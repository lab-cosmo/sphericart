#ifndef SPHERICART_TORCH_AUTOGRAD_HPP
#define SPHERICART_TORCH_AUTOGRAD_HPP

#include <vector>
#include <torch/torch.h>

namespace sphericart_torch {

class SphericartAutograd : public torch::autograd::Function<SphericartAutograd> {
  public:
    template <typename C>
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        C& calculator,
        torch::Tensor xyz,
        bool do_gradients,
        bool do_hessians
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
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

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_2_outputs
    );
};

} // namespace sphericart_torch

#endif
