#include "sphericart/torch_cuda_wrapper.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/torch.h>

#define _SPHERICART_INTERNAL_IMPLEMENTATION // gives us access to
                                            // templates/macros
#include "cuda_base.hpp"
#include "sphericart.hpp"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DTYPE(x, y)                                                                     \
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x " and " #y " must have the same dtype.")

#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

/*
    Torch wrapper for the CUDA kernel backwards pass.
*/
torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad, void* stream
) {

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error("internal error: CUDA version called on non-CUDA tensor");
    }

    auto xyz_grad = torch::Tensor();
    if (xyz.requires_grad()) {
        xyz_grad = torch::empty_like(xyz);

        AT_DISPATCH_FLOATING_TYPES(
            xyz.type(), "spherical_harmonics_backward_cuda", ([&] {
                sphericart::cuda::spherical_harmonics_backward_cuda_base<scalar_t>(
                    dsph.data_ptr<scalar_t>(),
                    sph_grad.data_ptr<scalar_t>(),
                    dsph.size(0),
                    sph_grad.size(1),
                    xyz_grad.data_ptr<scalar_t>(),
                    stream
                );
            })
        );
    }
    // synchronization happens within spherical_harmonics_backward_cuda_base
    return xyz_grad;
}