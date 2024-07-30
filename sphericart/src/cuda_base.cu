#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

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
    Wrapper to launch the CUDA kernel. Returns a vector containing the spherical
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

/*
template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t* __restrict__ xyz,
    const int nedges,
    const scalar_t* __restrict__ prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* __restrict__ sph,
    scalar_t* __restrict__ dsph,
    scalar_t* __restrict__ ddsph,
    void* cuda_stream
) {

    int n_total = (l_max + 1) * (l_max + 1);

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    dim3 block_dim(find_num_blocks(nedges, GRID_DIM_Y));

    size_t total_buff_size =
        total_buffer_size(l_max, GRID_DIM_X, GRID_DIM_Y, sizeof(scalar_t), gradients, hessian);

    spherical_harmonics_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size, cstream>>>(
        const_cast<scalar_t*>(xyz), nedges, const_cast<scalar_t*>(prefactors), nprefactors, l_max,
n_total, gradients, hessian, normalize, sph, dsph, ddsph
    );

    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaStreamSynchronize(cstream));
} */

// Function to load CUDA source code from a file
std::string load_cuda_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    scalar_t* xyz,
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

    std::vector<char> g_ptxCode;
    CUmodule g_module = nullptr;
    std::string g_last_kernel_name;
    nvrtcProgram prog;

    std::vector<std::string> kernel_name_vec = {
        "spherical_harmonics_kernel<double>", "spherical_harmonics_kernel<float>"
    };

    if (g_ptxCode.empty()) {
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

        /*std::cout << "mangled name: " << name << std::endl;
        std::cout << "PTX code size: " << ptxSize << " (bytes: " << ptxSize * sizeof(char) << ")"
                  << std::endl; */

        CUdevice cuDevice;
        CUcontext context;
        CUmodule module;

        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&context, 0, cuDevice);
        CUresult cuResult = cuModuleLoadDataEx(&module, g_ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to load PTX code into CUDA module" << std::endl;
            exit(1);
        }

        g_module = module;
    }

    const char* kernel_name =
        sizeof(scalar_t) == 8 ? kernel_name_vec[0].c_str() : kernel_name_vec[1].c_str();

    const char* name;

    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernel_name, &name));

    CUfunction kernel;

    cuModuleGetFunction(&kernel, g_module, name);

    int n_total = (l_max + 1) * (l_max + 1);

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    dim3 block_dim(find_num_blocks(nedges, GRID_DIM_Y));

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

    cuLaunchKernel(
        kernel,
        block_dim.x,
        block_dim.y,
        block_dim.z, // grid dim
        grid_dim.x,
        grid_dim.y,
        grid_dim.z, // block dim
        total_buff_size,
        cstream,
        args, // arguments
        0
    );

    // spherical_harmonics_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size, cstream>>>(
    //     const_cast<scalar_t*>(xyz), nedges, const_cast<scalar_t*>(prefactors), nprefactors,
    //     l_max, n_total, gradients, hessian, normalize, sph, dsph, ddsph
    //);
    cuCtxSynchronize();

    // CUDA_CHECK_KERNEL();

    // CUDA_CHECK(cudaStreamSynchronize(cstream));
}

template void sphericart::cuda::spherical_harmonics_cuda_base<float>(
    float* xyz,
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
    double* xyz,
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

/*
    CUDA kernel to computes the backwards pass for autograd.
*/
template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ dsph,
    const scalar_t* __restrict__ sph_grad,
    size_t nedges,
    size_t n_total,
    scalar_t* __restrict__ xyz_grad
) {

    size_t edge_idx = blockIdx.x * blockDim.y + threadIdx.y;

    int spatial = blockIdx.y;

    scalar_t sum = 0.0;

    if (edge_idx < nedges) {
        // for (int j = threadIdx.x; j < sph_grad.size(1); j += blockDim.x) {
        for (int j = threadIdx.x; j < n_total; j += blockDim.x) {

            // sum += dsph[edge_idx][spatial][j] * sph_grad[edge_idx][j];
            sum += dsph[edge_idx * 3 * n_total + spatial * n_total + j] *
                   sph_grad[edge_idx * n_total + j];
        }
    }

    __syncthreads();

    // reduce across the sub-warp
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (edge_idx < nedges) {
        if (threadIdx.x == 0) {
            // xyz_grad[sample_idx][spatial] = sum;
            xyz_grad[edge_idx * 3 + spatial] = sum;
        }
    }
}

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    const scalar_t* __restrict__ dsph,
    const scalar_t* __restrict__ sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* __restrict__ xyz_grad,
    void* cuda_stream
) {
    dim3 grid_dim(4, 32);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 block_dim(find_num_blocks(nedges, 32), 3);

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    backward_kernel<scalar_t>
        <<<block_dim, grid_dim, 0, cstream>>>(dsph, sph_grad, nedges, ntotal, xyz_grad);

    CUDA_CHECK_KERNEL();

    cudaStreamSynchronize(cstream);
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    const float* __restrict__ dsph,
    const float* __restrict__ sph_grad,
    const int nedges,
    const int ntotal,
    float* __restrict__ xyz_grad,
    void* cuda_stream
);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    const double* __restrict__ dsph,
    const double* __restrict__ sph_grad,
    const int nedges,
    const int ntotal,
    double* __restrict__ xyz_grad,
    void* cuda_stream
);
