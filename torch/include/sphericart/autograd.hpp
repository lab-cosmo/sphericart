#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <torch/data.h>
#include <torch/autograd.h>

namespace sphericart {

class SphericalHarmonicsAutograd : public torch::autograd::Function<SphericalHarmonicsAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int64_t l_max,
        torch::Tensor xyz,
        bool normalize
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};


}

#endif
