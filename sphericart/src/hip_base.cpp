#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <hip/hip_runtime.h>

#include "sphericart_impl.cuh"

#include "hip_base.hpp"
#include "hiprtc_lite.hpp"

#define HARDCODED_LMAX 1

// The kernel source is embedded as a byte array with a trailing 0x00 generated
// by make_includeable in CMakeLists.txt, matching the representation used in
// vesin and by the CUDA backend. The CUDA-dialect source is compiled at
// runtime by hipRTC (the mechanism ported from symmetrix-xl's HIP backend);
// hipCC/hipRTC accept CUDA-dialect kernels unchanged.
static const unsigned char HIP_CODE_DATA[] = {
#include "generated/wrapped_sphericart_impl.cu"
};
static const char* HIP_CODE = reinterpret_cast<const char*>(HIP_CODE_DATA);

static void check_hip(hipError_t status, const std::string& what) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            what + " failed: " + hipGetErrorString(status)
        );
    }
}

/*
    Computes the total amount of shared memory space required by
   spherical_harmonics_kernel. Mirrors the sizing in cuda_base.cpp.
*/
static size_t total_buffer_size(
    size_t l_max, size_t GRID_DIM_Y, size_t dtype_size, bool requires_grad, bool requires_hessian
) {
    int nl =
        std::max(static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)), 2 * l_max + 1);

    size_t total_buff_size = 0;

    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size;  // buffer_c
    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size;  // buffer_s
    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size;  // buffer_twomz
    total_buff_size += (l_max + 1) * (l_max + 2) * dtype_size; // buffer_prefactors
    total_buff_size += GRID_DIM_Y * nl * dtype_size;           // buffer_sph_out

    if (requires_grad) {
        total_buff_size += 3 * GRID_DIM_Y * nl * dtype_size; // buffer_sph_derivs
    }

    if (requires_hessian) {
        total_buff_size += 9 * GRID_DIM_Y * nl * dtype_size; // buffer_sph_hessian
    }

    return total_buff_size;
}

// Name expression for the template kernels, e.g.
// "spherical_harmonics_kernel<float>". The explicit float/double
// instantiations live at the bottom of sphericart_impl.cu.
template <typename scalar_t>
static std::string template_kernel_name(const std::string& fn_name) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        return fn_name + "<float>";
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        return fn_name + "<double>";
    } else {
        static_assert(
            sizeof(scalar_t) == 0, "sphericart HIP kernels only support float and double"
        );
    }
}

/*
    Runtime-compiled kernel handle for one device.

    Mirrors gpulite's per-device KernelFactory on the CUDA side: the module is
    compiled with hipRTC for the device's own GCN architecture (queried at
    runtime from hipGetDeviceProperties, following symmetrix-xl), loaded with
    hipModuleLoadData, and its kernels resolved through the lowered (mangled)
    names of the requested name expressions.
*/
struct HipKernelModule {
    hipModule_t module = nullptr;
    std::unordered_map<std::string, hipFunction_t> functions;
    int default_shared_memory_per_block = -1;
    int optin_shared_memory_per_block = -1;
    std::unordered_map<const void*, bool> shared_memory_opted_in;
};

static HipKernelModule& device_kernel_cache(int device) {
    static std::mutex cache_mutex;
    static std::unordered_map<int, HipKernelModule> cache;

    std::lock_guard<std::mutex> lock(cache_mutex);
    return cache[device];
}

static hipFunction_t get_hip_kernel(int device, const std::string& kernel_expression) {
    auto& entry = device_kernel_cache(device);
    static std::mutex compile_mutex;
    std::lock_guard<std::mutex> lock(compile_mutex);

    auto found = entry.functions.find(kernel_expression);
    if (found != entry.functions.end()) {
        return found->second;
    }

    // Resolve the target architecture at runtime, like symmetrix-xl does for
    // its JIT modules: gcnArchName may carry feature suffixes
    // ("gfx1151:sramecc+:xnack-"), which are not valid --gpu-architecture
    // values and must be stripped.
    hipDeviceProp_t properties;
    check_hip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    std::string gcn_arch(properties.gcnArchName);
    auto feature_suffix = gcn_arch.find(':');
    if (feature_suffix != std::string::npos) {
        gcn_arch.erase(feature_suffix);
    }

    auto compilation = sphericart::hip::compile_hip_with_hiprtc(
        std::string(HIP_CODE),
        "wrapped_sphericart_impl.cu",
        {kernel_expression},
        {"--std=c++17", "--gpu-architecture=" + gcn_arch}
    );

    // hipModuleLoadData attaches the module to the current device context
    int current_device = 0;
    check_hip(hipGetDevice(&current_device), "hipGetDevice");
    if (current_device != device) {
        check_hip(hipSetDevice(device), "hipSetDevice");
    }

    check_hip(hipModuleLoadData(&entry.module, compilation.code.data()), "hipModuleLoadData");

    auto lowered = compilation.lowered_names.find(kernel_expression);
    if (lowered == compilation.lowered_names.end()) {
        throw std::runtime_error(
            "hipRTC did not lower the kernel name expression " + kernel_expression
        );
    }

    hipFunction_t function = nullptr;
    check_hip(
        hipModuleGetFunction(&function, entry.module, lowered->second.c_str()),
        "hipModuleGetFunction(" + kernel_expression + ")"
    );

    if (current_device != device) {
        check_hip(hipSetDevice(current_device), "hipSetDevice");
    }

    entry.functions[kernel_expression] = function;
    return function;
}

/*
    Opt the kernel into more than the default dynamic shared memory when the
    launch needs it (the HIP analog of gpulite's checkAndAdjustSharedMem).
*/
static void ensure_shared_memory(int device, hipFunction_t function, unsigned int shared_bytes) {
    auto& entry = device_kernel_cache(device);
    static std::mutex attribute_mutex;
    std::lock_guard<std::mutex> lock(attribute_mutex);

    if (entry.default_shared_memory_per_block < 0) {
        int default_smem = 0;
        check_hip(
            hipDeviceGetAttribute(
                &default_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, device),
            "hipDeviceGetAttribute(MaxSharedMemoryPerBlock)"
        );
        int optin_smem = 0;
        check_hip(
            hipDeviceGetAttribute(
                &optin_smem, hipDeviceAttributeSharedMemPerBlockOptin, device),
            "hipDeviceGetAttribute(SharedMemPerBlockOptin)"
        );
        entry.default_shared_memory_per_block = default_smem;
        entry.optin_shared_memory_per_block = optin_smem;
    }

    if (shared_bytes <= static_cast<unsigned int>(entry.default_shared_memory_per_block)) {
        return;
    }

    auto& opted_in = entry.shared_memory_opted_in[function];
    if (opted_in) {
        return;
    }

    check_hip(
        hipFuncSetAttribute(
            function, hipFuncAttributeMaxDynamicSharedMemorySize,
            entry.optin_shared_memory_per_block),
        "hipFuncSetAttribute(MaxDynamicSharedMemorySize)"
    );
    opted_in = true;
}

/*
    Wrapper to compile and launch the HIP kernel.

    GRID_DIM_X is the number of threads to launch in the x dimension. Used to
    parallelize over the sample dimension. GRID_DIM_Y is the number of threads
    to launch in the y dimension. Used only to improve memory throughput on
    reads and writes.

    hip_stream should be of type (void *), therefore if you want to pass in
    a hipStream_t, first do void * stream_ptr = reinterpret_cast<void *>
    (stream);
*/
template <typename scalar_t>
void sphericart::hip::spherical_harmonics_hip_base(
    const scalar_t* xyz,
    const int nedges,
    const scalar_t* prefactors,
    const int nprefactors,
    const int l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    void* hip_stream,
    bool synchronize
) {
    int n_total = (l_max + 1) * (l_max + 1);
    auto find_num_blocks = [](int x, int bdim) { return x / bdim + (x % bdim != 0); };

    int device;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    hipFunction_t function =
        get_hip_kernel(device, template_kernel_name<scalar_t>("spherical_harmonics_kernel"));

    unsigned int shared_bytes = static_cast<unsigned int>(
        total_buffer_size(l_max, GRID_DIM_Y, sizeof(scalar_t), gradients, hessian));
    ensure_shared_memory(device, function, shared_bytes);

    void* parameters[] = {
        const_cast<scalar_t**>(&xyz),
        const_cast<int*>(&nedges),
        const_cast<scalar_t**>(&prefactors),
        const_cast<int*>(&nprefactors),
        const_cast<int*>(&l_max),
        const_cast<int*>(&n_total),
        const_cast<bool*>(&gradients),
        const_cast<bool*>(&hessian),
        const_cast<bool*>(&normalize),
        &sph,
        &dsph,
        &ddsph,
    };

    check_hip(
        hipModuleLaunchKernel(
            function,
            static_cast<unsigned int>(find_num_blocks(nedges, static_cast<int>(GRID_DIM_Y))),
            1,
            1,
            static_cast<unsigned int>(GRID_DIM_X),
            static_cast<unsigned int>(GRID_DIM_Y),
            1,
            shared_bytes,
            reinterpret_cast<hipStream_t>(hip_stream),
            parameters,
            nullptr),
        "hipModuleLaunchKernel(spherical_harmonics_kernel)");
    if (synchronize) {
        check_hip(
            hipStreamSynchronize(reinterpret_cast<hipStream_t>(hip_stream)),
            "hipStreamSynchronize");
    }
}

template void sphericart::hip::spherical_harmonics_hip_base<float>(
    const float* xyz,
    const int nedges,
    const float* prefactors,
    const int nprefactors,
    const int l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    float* sph,
    float* dsph,
    float* ddsph,
    void* hip_stream,
    bool synchronize
);

template void sphericart::hip::spherical_harmonics_hip_base<double>(
    const double* xyz,
    const int nedges,
    const double* prefactors,
    const int nprefactors,
    const int l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    double* sph,
    double* dsph,
    double* ddsph,
    void* hip_stream,
    bool synchronize
);

template <typename scalar_t>
void sphericart::hip::spherical_harmonics_backward_hip_base(
    const scalar_t* dsph,
    const scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* hip_stream
) {
    auto find_num_blocks = [](int x, int bdim) { return x / bdim + (x % bdim != 0); };

    int device;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    hipFunction_t function =
        get_hip_kernel(device, template_kernel_name<scalar_t>("backward_kernel"));

    void* parameters[] = {
        const_cast<scalar_t**>(&dsph),
        const_cast<scalar_t**>(&sph_grad),
        const_cast<int*>(&nedges),
        const_cast<int*>(&ntotal),
        &xyz_grad,
    };

    check_hip(
        hipModuleLaunchKernel(
            function,
            static_cast<unsigned int>(find_num_blocks(nedges, 32)),
            3,
            1,
            4,
            32,
            1,
            0,
            reinterpret_cast<hipStream_t>(hip_stream),
            parameters,
            nullptr),
        "hipModuleLaunchKernel(backward_kernel)");
}

template void sphericart::hip::spherical_harmonics_backward_hip_base<float>(
    const float* dsph,
    const float* sph_grad,
    const int nedges,
    const int ntotal,
    float* xyz_grad,
    void* hip_stream
);

template void sphericart::hip::spherical_harmonics_backward_hip_base<double>(
    const double* dsph,
    const double* sph_grad,
    const int nedges,
    const int ntotal,
    double* xyz_grad,
    void* hip_stream
);
