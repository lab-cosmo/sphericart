#include "sphericart.hpp"
#include "sphericart/torch_cuda_wrapper.hpp"
#include <torch/torch.h>

/*
    Torch wrapper for the CUDA kernel forwards pass.
*/
std::vector<torch::Tensor> sphericart_torch::spherical_harmonics_cuda(
    torch::Tensor xyz, torch::Tensor prefactors, int64_t l_max, bool normalize,
    int64_t GRID_DIM_X, int64_t GRID_DIM_Y, bool gradients, bool hessian) {

    throw std::runtime_error(
        "sphericart_torch was not compiled with CUDA support");

    return {torch::Tensor()};
}

/*
    Torch wrapper for the CUDA kernel backwards pass.
*/
torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad) {

    throw std::runtime_error(
        "sphericart_torch was not compiled with CUDA support");

    return {torch::Tensor()};
}