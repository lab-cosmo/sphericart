#ifndef SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP
#define SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP

#include <ATen/Tensor.h>
#include <torch/torch.h>
#include <vector>

namespace sphericart_torch {

at::Tensor spherical_harmonics_backward_cuda(
    at::Tensor xyz, at::Tensor dsph, at::Tensor sph_grad, void* cuda_stream
);

} // namespace sphericart_torch

#endif