#include <stdexcept>

#include "cuda_base.hpp"

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t* /*xyz*/,
    const int /*nedges*/,
    const scalar_t* /*prefactors*/,
    const int /*nprefactors*/,
    const int64_t /*l_max*/,
    const bool /*normalize*/,
    const int64_t /*GRID_DIM_X*/,
    const int64_t /*GRID_DIM_Y*/,
    const bool /*gradients*/,
    const bool /*hessian*/,
    scalar_t* /*sph*/,
    scalar_t* /*dsph*/,
    scalar_t* /*ddsph*/,
    void* /*cuda_stream*/
) {

    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

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

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    const scalar_t* /*dsph*/,
    const scalar_t* /*sph_grad*/,
    const int /*nedges*/,
    const int /*ntotal*/,
    scalar_t* /*xyz_grad*/,
    void* /*cuda_stream*/
) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
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

int sphericart::cuda::adjust_shared_memory(size_t, int64_t, int64_t, int64_t, bool, bool, int64_t) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
    return -1;
}
