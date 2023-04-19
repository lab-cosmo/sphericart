#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <torch/data.h>
#include <torch/autograd.h>

namespace sphericart_torch {

class SphericalHarmonics;

class SphericalHarmonicsAutograd : public torch::autograd::Function<SphericalHarmonicsAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        SphericalHarmonics& calculator,
        torch::Tensor xyz,
        bool gradients
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

} // namespace sphericart_torch

#endif
