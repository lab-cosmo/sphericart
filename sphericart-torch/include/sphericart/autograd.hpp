#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <torch/autograd.h>
#include <torch/data.h>
#include <ATen/Tensor.h>
#include <vector>

namespace sphericart_torch {

std::vector<at::Tensor> spherical_harmonics_cuda(at::Tensor xyz, at::Tensor prefactors, int64_t l_max,
                         bool normalize, int64_t GRID_DIM_X, int64_t GRID_DIM_Y,
                         bool gradients, bool hessians);

at::Tensor spherical_harmonics_backward_cuda(at::Tensor xyz, at::Tensor dsph,
                                             at::Tensor sph_grad); 

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
