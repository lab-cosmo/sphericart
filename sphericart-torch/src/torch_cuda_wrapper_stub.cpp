#include "sphericart.hpp"
#include "sphericart/torch_cuda_wrapper.hpp"
#include <torch/torch.h>

/*
    Torch wrapper for the CUDA kernel backwards pass.
*/
torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad, int64_t cuda_stream
) {

    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");

    return {torch::Tensor()};
}