#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <ATen/Tensor.h>
#include <torch/autograd.h>
#include <torch/data.h>
#include <vector>

namespace sphericart_torch {

class SphericalHarmonics;

class SphericartAutograd : public torch::autograd::Function<SphericartAutograd> {
  public:
    template <typename C>
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        C& calculator,
        torch::Tensor xyz,
        bool do_gradients,
        bool do_hessians
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs
    );
};

class SphericartAutogradBackward : public torch::autograd::Function<SphericartAutogradBackward> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor grad_outputs,
        torch::Tensor xyz,
        std::vector<torch::Tensor> saved_variables
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_2_outputs
    );
};

} // namespace sphericart_torch

#endif
