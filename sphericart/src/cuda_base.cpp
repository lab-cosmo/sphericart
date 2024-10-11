#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include "cuda_cache.hpp"
#include "cuda_base.hpp"

#define HARDCODED_LMAX 1

/*
    Computes the total amount of shared memory space required by
   spherical_harmonics_kernel.

    For lmax <= HARCODED_LMAX, we need to store all (HARDCODED_LMAX + 1)**2
   scalars in shared memory. For lmax > HARDCODED_LMAX, we only need to store
   each spherical harmonics vector per sample in shared memory.
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
    const scalar_t* xyz,
    const int nedges,
    const scalar_t* prefactors,
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
#include "generated/wrapped_sphericart_impl.cu"
        ;

    int n_total = (l_max + 1) * (l_max + 1);
    dim3 block_dim(GRID_DIM_X, GRID_DIM_Y);
    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);
    dim3 grid_dim(find_num_blocks(nedges, GRID_DIM_Y));

    size_t smem_size = total_buffer_size(l_max, GRID_DIM_Y, sizeof(scalar_t), gradients, hessian);

    scalar_t* _xyz = const_cast<scalar_t*>(xyz);
    scalar_t* _prefactors = const_cast<scalar_t*>(prefactors);
    int _nprefactors = nprefactors;
    int _lmax = l_max;
    int _n_total = n_total;
    int _nedges = nedges;
    bool _gradients = gradients;
    bool _hessian = hessian;
    bool _normalize = normalize;

    std::vector<void*> args = {
        &_xyz,
        &_nedges,
        &_prefactors,
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
        kernel_name, std::string(CUDA_CODE), "wrapped_sphericart_impl.cu", {"--std=c++17"}
    );

    kernel->launch(grid_dim, block_dim, smem_size, cstream, args);
}

template void sphericart::cuda::spherical_harmonics_cuda_base<float>(
    const float* xyz,
    const int nedges,
    const float* prefactors,
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
    const double* prefactors,
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
    const scalar_t* dsph,
    const scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* cuda_stream
) {
    static const char* CUDA_CODE =
#include "generated/wrapped_sphericart_impl.cu"
        ;

    std::string kernel_name = getKernelName<scalar_t>("backward_kernel");

    auto& kernel_factory = KernelFactory::instance();

    dim3 block_dim(4, 32);
    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    dim3 grid_dim(find_num_blocks(nedges, 32), 3);
    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    int _n_total = ntotal;
    int _nedges = nedges;

    std::vector<void*> args = {&dsph, &sph_grad, &_nedges, &_n_total, &xyz_grad};

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_sphericart_impl.cu", {"--std=c++17"}
    );

    kernel->launch(grid_dim, block_dim, 0, cstream, args);
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    const float* dsph,
    const float* sph_grad,
    const int nedges,
    const int ntotal,
    float* xyz_grad,
    void* cuda_stream
);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    const double* dsph,
    const double* sph_grad,
    const int nedges,
    const int ntotal,
    double* xyz_grad,
    void* cuda_stream
);
