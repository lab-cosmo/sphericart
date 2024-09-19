#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <typeinfo>

#include <dlfcn.h>
#include <cstring>

#include <cstdlib> //std::getenv

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

#include "cuda_cache.hpp"
#include "cuda_base.hpp"

/* MASK used for warp reductions */
#define FULL_MASK 0xffffffff

#define HARDCODED_LMAX 1

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

/*
hack to obtain sphericart base directory from
sphericart.so/sphericart_jax.so/sphericart_torch.so path
linux only - windows can be done as well
*/

/// home/nick/miniconda3/envs/sphericart/lib/python3.12/site-packages/sphericart/torch/lib/sphericart/package_data/sphericart_impl.cu
std::string getDirRelativeToLib(std::string directoryName) {
    Dl_info dl_info;
    if (dladdr((void*)getDirRelativeToLib, &dl_info) && dl_info.dli_fname) {
        std::string libpath = std::string(dl_info.dli_fname);
        // Find the last occurrence of 'sphericart' in the path
        std::string base_name = directoryName + "/";
        size_t start_pos =
            libpath.rfind(base_name); // Use rfind to start from the end of the string

        if (start_pos == std::string::npos) {
            return ""; // 'sphericart' not found
        }

        // Find the last directory separator before the 'sphericart' occurrence
        size_t end_pos = libpath.find_last_of("/\\", start_pos);
        if (end_pos == std::string::npos) {
            return ""; // No directory separator found
        }

        // Extract the base directory path
        return libpath.substr(0, end_pos + 1) + directoryName;
    }
    return "";
}

/*
    Wrapper to compile and launch the CUDA kernel. outputs a vector containing the spherical
   harmonics and their gradients if required to sph, dsph and ddsph pointers.

    GRID_DIM_X is the number of threads to launch in the x dimension. Used to
   parallelize over the sample dimension. GRID_DIM_Y is the number of threads to
   launch in the y dimension. Used only to improve memory throughput on reads
   and writes.

    Total number of threads used is GRID_DIM_X * GRID_DIM_Y.

    cuda_stream should be of type (void *), therefore if you want to pass in
    a cudaStream_t, first do void * stream_ptr = reinterpret_cast<void *>
   (stream);
*/
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

    static const char* CUDA_CODE =
#include "generated/wrapped_sphericart_impl.cuh"
        ;

    int n_total = (l_max + 1) * (l_max + 1);
    dim3 block_dim(GRID_DIM_X, GRID_DIM_Y);
    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);
    dim3 grid_dim(find_num_blocks(nedges, GRID_DIM_Y));

    size_t smem_size =
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

    std::string kernel_name = getKernelName<scalar_t>("spherical_harmonics_kernel");
    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name,
        std::string(CUDA_CODE),
        "sphericart_impl.cu",
        {"--define-macro=CUDA_DEVICE_PREFIX=__device__"}
    );

    kernel->launch(grid_dim, block_dim, smem_size, cstream, args, 12);
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

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    scalar_t* dsph,
    scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* cuda_stream
) {

    static const char* CUDA_CODE =
#include "generated/wrapped_sphericart_impl.cuh"
        ;
    std::string kernel_name = getKernelName<scalar_t>("backward_kernel");

    auto& kernel_factory = KernelFactory::instance();

    dim3 block_dim(4, 32);
    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    dim3 grid_dim(find_num_blocks(nedges, 32), 3);
    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    int _n_total = ntotal;
    int _nedges = nedges;

    void* args[] = {&dsph, &sph_grad, &_nedges, &_n_total, &xyz_grad};

    CachedKernel* kernel = kernel_factory.create(
        kernel_name,
        std::string(CUDA_CODE),
        "sphericart_impl.cu",
        {"--define-macro=CUDA_DEVICE_PREFIX=__device__"}
    );

    kernel->launch(grid_dim, block_dim, 0, cstream, args, 5);
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    float* dsph, float* sph_grad, const int nedges, const int ntotal, float* xyz_grad, void* cuda_stream
);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    double* dsph, double* sph_grad, const int nedges, const int ntotal, double* xyz_grad, void* cuda_stream
);
