#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <ATen/Tensor.h>
#include <torch/autograd.h>
#include <torch/data.h>
#include <vector>

namespace sphericart_torch {

class SphericalHarmonics;

class SphericalHarmonicsAutograd
    : public torch::autograd::Function<SphericalHarmonicsAutograd> {
  public:
    static torch::autograd::variable_list
    forward(torch::autograd::AutogradContext *ctx,
            SphericalHarmonics &calculator, torch::Tensor xyz,
            bool do_gradients, bool do_hessians);

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs);
};

class SphericalHarmonicsAutogradBackward
    : public torch::autograd::Function<SphericalHarmonicsAutogradBackward> {
  public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor grad_outputs, torch::Tensor xyz,
                                 std::vector<torch::Tensor> saved_variables);

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_2_outputs);
};

} // namespace sphericart_torch

#endif
