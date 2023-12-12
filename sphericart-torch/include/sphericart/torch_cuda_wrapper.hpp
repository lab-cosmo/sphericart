#ifndef SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP
#define SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP

#include <ATen/Tensor.h>
#include <torch/torch.h>
#include <vector>

namespace sphericart_torch {

std::vector<at::Tensor>
spherical_harmonics_cuda(at::Tensor xyz, at::Tensor prefactors, int64_t l_max,
                         bool normalize, int64_t GRID_DIM_X, int64_t GRID_DIM_Y,
                         bool gradients, bool hessians);

at::Tensor spherical_harmonics_backward_cuda(at::Tensor xyz, at::Tensor dsph,
                                             at::Tensor sph_grad);

torch::Tensor prefactors_cuda(int64_t l_max, at::ScalarType dtype);

} // namespace sphericart_torch

#endif