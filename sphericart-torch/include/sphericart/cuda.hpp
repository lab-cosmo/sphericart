#ifndef SPHERICART_TORCH_CUDA_HPP
#define SPHERICART_TORCH_CUDA_HPP

#include <ATen/Tensor.h>
#include <vector>
#include <cuda.h>

namespace sphericart_torch {

std::vector<at::Tensor>
spherical_harmonics_cuda(at::Tensor xyz, at::Tensor prefactors, int64_t l_max,
                         bool normalize, int64_t GRID_DIM_X, int64_t GRID_DIM_Y,
                         bool gradients, bool hessians);

at::Tensor spherical_harmonics_backward_cuda(at::Tensor xyz, at::Tensor dsph,
                                             at::Tensor sph_grad);

at::Tensor prefactors_cuda(int64_t l_max, at::ScalarType dtype);

bool adjust_cuda_shared_memory(
    size_t element_size, int64_t l_max, int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y, bool requires_grad, bool requires_hessian);



} // namespace sphericart_torch

#endif
