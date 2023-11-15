#ifndef SPHERICART_TORCH_CUDA_HPP
#define SPHERICART_TORCH_CUDA_HPP

namespace sphericart_torch {

template <typename scalar_t>
void spherical_harmonics_cuda_base(
    const scalar_t *__restrict__ xyz, const int nedges,
    const scalar_t *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool xyz_requires_grad,
    const bool gradients, const bool hessian, scalar_t *__restrict__ sph,
    scalar_t *__restrict__ dsph, scalar_t *__restrict__ ddsph);

template <typename scalar_t>
void spherical_harmonics_backward_cuda_base(
    const scalar_t * __restrict__ dsph,
    const scalar_t * __restrict__ sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t * __restrict__ xyz_grad);
                         

bool adjust_cuda_shared_memory(
    size_t element_size, int64_t l_max, int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y, bool requires_grad, bool requires_hessian);



} // namespace sphericart_torch

#endif
