#include <stdexcept>

#include "cuda_base.hpp"

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t *__restrict__ xyz, const int nedges,
    const scalar_t *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    scalar_t *__restrict__ sph, scalar_t *__restrict__ dsph,
    scalar_t *__restrict__ ddsph, void *cuda_stream) {

    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template void sphericart::cuda::spherical_harmonics_cuda_base<double>(
    const double *__restrict__ xyz, const int nedges,
    const double *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    double *__restrict__ sph, double *__restrict__ dsph,
    double *__restrict__ ddsph, void *cuda_stream);

template void sphericart::cuda::spherical_harmonics_cuda_base<float>(
    const float *__restrict__ xyz, const int nedges,
    const float *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    float *__restrict__ sph, float *__restrict__ dsph,
    float *__restrict__ ddsph, void *cuda_stream);

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    const scalar_t *__restrict__ dsph, const scalar_t *__restrict__ sph_grad,
    const int nedges, const int ntotal, scalar_t *__restrict__ xyz_grad,
    void *cuda_stream) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    const float *__restrict__ dsph, const float *__restrict__ sph_grad,
    const int nedges, const int ntotal, float *__restrict__ xyz_grad,
    void *cuda_stream);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    const double *__restrict__ dsph, const double *__restrict__ sph_grad,
    const int nedges, const int ntotal, double *__restrict__ xyz_grad,
    void *cuda_stream);

int sphericart::cuda::adjust_shared_memory(size_t, int64_t, int64_t, int64_t,
                                           bool, bool, int64_t) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
    return -1;
}