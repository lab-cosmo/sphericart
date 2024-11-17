#ifndef SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP
#define SPHERICART_TORCH_TORCH_CUDA_WRAPPER_HPP

#include <vector>
#include <torch/torch.h>

namespace sphericart_torch {

torch::Tensor spherical_harmonics_backward_cuda(
    torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad, int64_t stream = 0
);

} // namespace sphericart_torch

#endif
