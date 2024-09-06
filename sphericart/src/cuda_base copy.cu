#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

#include <string>
#include <typeinfo>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

#include "cuda_cache.hpp"
#include "cuda_base.hpp"
#include "sphericart_impl.cuh"

/* MASK used for warp reductions */
#define FULL_MASK 0xffffffff

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result)       \
                      << '\n';                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CUDA_SAFE_CALL(x)                                                                          \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            cuGetErrorName(result, &msg);                                                          \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n';                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define HARDCODED_LMAX 1

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(cudaStatus) << std::endl;                              \
            cudaDeviceReset();                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                        \
    do {                                                                                           \
        cudaDeviceSynchronize();                                                                   \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            fprintf(                                                                               \
                stderr,                                                                            \
                "CUDA error after kernel launch in %s at line %d: %s\n",                           \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                cudaGetErrorString(err)                                                            \
            );                                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

/*
    Computes the total amount of shared memory space required by
   spherical_harmonics_kernel.

    For lmax <= HARCODED_LMAX, we need to store all (HARDCODED_LMAX + 1)**2
   scalars in shared memory. For lmax > HARDCODED_LMAX, we only need to store
   each spherical harmonics vector per sample in shared memory.
*/
static size_t total_buffer_size(
    size_t l_max,
    size_t GRID_DIM_X,
    size_t GRID_DIM_Y,
    size_t dtype_size,
    bool requires_grad,
    bool requires_hessian
) {
    int nl = max(static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)), 2 * l_max + 1);

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

/*
    The default shared memory space on most recent NVIDIA cards is defaulted
   49152 bytes, regarldess if there is more available per SM. This method
   attempts to adjust the shared memory to fit the requested configuration if
   the kernel launch parameters exceeds the default 49152 bytes.
*/

int sphericart::cuda::adjust_shared_memory(
    size_t element_size,
    int64_t l_max,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool requires_grad,
    bool requires_hessian,
    int64_t current_shared_mem_alloc
) {

    int device_count;

    cudaGetDeviceCount(&device_count);

    int current_device;
    cudaGetDevice(&current_device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, current_device);

    auto required_buff_size = total_buffer_size(
        l_max, GRID_DIM_X, GRID_DIM_Y, element_size, requires_grad, requires_hessian
    );

    if (required_buff_size > current_shared_mem_alloc &&
        required_buff_size > (deviceProp.sharedMemPerBlock - deviceProp.reservedSharedMemPerBlock)) {

        if (required_buff_size > deviceProp.sharedMemPerBlockOptin) {
            return -1; // failure - need to adjust parameters
        }

        // broadcast changes to all visible GPUs
        for (int device = 0; device < device_count; device++) {

            CUDA_CHECK(cudaSetDevice(device));

            switch (element_size) {
            case 8:
                cudaFuncSetAttribute(
                    spherical_harmonics_kernel<double>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    required_buff_size
                );
                break;
            case 4:
                cudaFuncSetAttribute(
                    spherical_harmonics_kernel<float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    required_buff_size
                );
                break;
            }
        }

        CUDA_CHECK(cudaSetDevice(current_device));

        return required_buff_size;

    } else {
        return (current_shared_mem_alloc >
                (deviceProp.sharedMemPerBlock - deviceProp.reservedSharedMemPerBlock))
                   ? current_shared_mem_alloc
                   : (deviceProp.sharedMemPerBlock - deviceProp.reservedSharedMemPerBlock);
    }
}

/*
    Wrapper to compile and launch the CUDA kernel. Returns a vector containing the spherical
   harmonics and their gradients if required, otherwise returns the spherical
   harmonics and an empty tensor.

    GRID_DIM_X is the number of threads to launch in the x dimension. Used to
   parallelize over the sample dimension. GRID_DIM_Y is the number of threads to
   launch in the y dimension. Used only to improve memory throughput on reads
   and writes.

    Total number of threads used is GRID_DIM_X * GRID_DIM_Y.

    cuda_stream should be of type (void *), therefore if you want to pass in
    a cudaStream_t, first do void * stream_ptr = reinterpret_cast<void *>
   (stream);
*/

template <typename T> std::string getKernelNameForType(const std::string& fn_name) {
    std::string type_name = typeid(T).name();

// Demangle the type name if necessary
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled_name = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
    if (status == 0 && demangled_name != nullptr) {
        type_name = demangled_name;
    }
    free(demangled_name);
#endif

    return fn_name + "<" + type_name + ">";
}

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t* xyz,
    const int nedges,
    scalar_t* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    void* cuda_stream
) {

    std::string kernel_name = getKernelNameForType<scalar_t>("spherical_harmonics_kernel");

    auto& cache_manager = CudaCacheManager::instance();

    CUcontext currentContext;
    // Get current context
    cuCtxGetCurrent(&currentContext);

    if (!cache_manager.hasKernel(kernel_name)) {

        std::string kernel_code = load_cuda_source(SPHERICART_CUDA_SRC_PATH);

        nvrtcProgram prog;
        nvrtcResult result =
            nvrtcCreateProgram(&prog, kernel_code.c_str(), "sphericart_impl.cu", 0, nullptr, nullptr);

        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_name.c_str()));

        const char* opts[] = {
            "--include-path=" SPHERICART_INCLUDE_PATH, "--define-macro=CUDA_DEVICE_PREFIX=__device__"
        };

        nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
            std::cerr << log << std::endl;
            std::cerr << "Failed to compile CUDA program" << std::endl;
            exit(1);
        }

        // Get PTX code
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptxCode(ptxSize);
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptxCode.data()));

        CUdevice cuDevice;
        CUcontext context;
        CUmodule module;
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        CUresult cuResult = cuModuleLoadDataEx(&module, ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to load PTX code into CUDA module" << std::endl;
            exit(1);
        }

        const char* lowered_name;
        nvrtcGetLoweredName(prog, kernel_name.c_str(), &lowered_name);
        CUfunction kernel;
        cuModuleGetFunction(&kernel, module, lowered_name);
        cache_manager.cacheKernel(kernel_name, module, kernel, currentContext);

        // CUDA_SAFE_CALL(cuModuleUnload(module));
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    }

    if (cache_manager.hasKernel(kernel_name)) {

        CachedKernel kernel = cache_manager.getKernel(kernel_name);

        int n_total = (l_max + 1) * (l_max + 1);
        dim3 block_dim(GRID_DIM_X, GRID_DIM_Y);
        auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
        cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);
        dim3 grid_dim(find_num_blocks(nedges, GRID_DIM_Y));

        size_t total_buff_size =
            total_buffer_size(l_max, GRID_DIM_X, GRID_DIM_Y, sizeof(scalar_t), gradients, hessian);

        int _nprefactors = nprefactors;
        int _lmax = l_max;
        int _n_total = n_total;
        int _nedges = nedges;
        bool _gradients = gradients;
        bool _hessian = hessian;
        bool _normalize = normalize;

        void* args[] = {
            &xyz,
            &_nedges,
            &prefactors,
            &_nprefactors,
            &_lmax,
            &_n_total,
            &_gradients,
            &_hessian,
            &_normalize,
            &sph,
            &dsph,
            &ddsph
        };

        kernel.launch(grid_dim, block_dim, total_buff_size, cstream, args);

        cuCtxSynchronize();
    }
}

template void sphericart::cuda::spherical_harmonics_cuda_base<float>(
    const float* xyz,
    const int nedges,
    float* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    float* sph,
    float* dsph,
    float* ddsph,
    void* cuda_stream
);

template void sphericart::cuda::spherical_harmonics_cuda_base<double>(
    const double* xyz,
    const int nedges,
    double* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    double* sph,
    double* dsph,
    double* ddsph,
    void* cuda_stream
);

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    scalar_t* dsph,
    scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* cuda_stream
) {

    static std::vector<char> g_ptxCode;
    static CUmodule g_module = nullptr;
    static nvrtcProgram prog;

    std::vector<std::string> kernel_name_vec = {"backward_kernel<double>", "backward_kernel<float>"};

    if (!g_module) {
        std::string kernel_code = load_cuda_source(SPHERICART_CUDA_SRC_PATH);

        nvrtcResult result =
            nvrtcCreateProgram(&prog, kernel_code.c_str(), "sphericart_impl.cu", 0, nullptr, nullptr);

        for (int i = 0; i < kernel_name_vec.size(); i++) {
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_name_vec[i].c_str()));
        }

        const char* opts[] = {
            "--include-path=" SPHERICART_INCLUDE_PATH, "--define-macro=CUDA_DEVICE_PREFIX=__device__"
        };
        nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);

        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
            char* log = new char[logSize];
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
            std::cout << log << '\n';
            delete[] log;
            std::cerr << "Failed to compile CUDA program" << std::endl;
            exit(1);
        }

        // Get PTX code
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        g_ptxCode.resize(ptxSize);
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, g_ptxCode.data()));

        CUdevice cuDevice;
        CUcontext context;
        CUmodule module;

        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&context, 0, cuDevice);
        CUresult cuResult = cuModuleLoadDataEx(&g_module, g_ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to load PTX code into CUDA module" << std::endl;
            exit(1);
        }
    }

    const char* kernel_name =
        sizeof(scalar_t) == 8 ? kernel_name_vec[0].c_str() : kernel_name_vec[1].c_str();

    const char* name;

    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernel_name, &name));

    static CUfunction kernel;

    cuModuleGetFunction(&kernel, g_module, name);

    dim3 block_dim(4, 32);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 grid_dim(find_num_blocks(nedges, 32), 3);

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    int _n_total = ntotal;
    int _nedges = nedges;

    void* args[] = {&dsph, &sph_grad, &_nedges, &_n_total, &xyz_grad};

    cuLaunchKernel(
        kernel,
        grid_dim.x,
        grid_dim.y,
        grid_dim.z, // grid dim
        block_dim.x,
        block_dim.y,
        block_dim.z, // block dim
        0,
        cstream,
        args, // arguments
        0
    );

    cuCtxSynchronize();
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    float* dsph, float* sph_grad, const int nedges, const int ntotal, float* xyz_grad, void* cuda_stream
);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    double* dsph, double* sph_grad, const int nedges, const int ntotal, double* xyz_grad, void* cuda_stream
);
