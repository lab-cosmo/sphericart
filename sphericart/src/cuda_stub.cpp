#include <stdexcept>

#include "cuda.hpp"

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t *__restrict__ xyz, const int nedges,
    const scalar_t *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool xyz_requires_grad,
    const bool gradients, const bool hessian, scalar_t *__restrict__ sph,
    scalar_t *__restrict__ dsph, scalar_t *__restrict__ ddsph) {

    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    const scalar_t *__restrict__ dsph, const scalar_t *__restrict__ sph_grad,
    const int nedges, const int ntotal, scalar_t *__restrict__ xyz_grad) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

bool sphericart::cuda::adjust_cuda_shared_memory(size_t, int64_t, int64_t,
                                                 int64_t, bool, bool) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}