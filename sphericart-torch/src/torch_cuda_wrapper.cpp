#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "sphericart/torch_cuda_wrapper.hpp"
#include "sphericart/cuda.hpp"

#define _SPHERICART_INTERNAL_IMPLEMENTATION // gives us access to templates/macros
#include "sphericart.hpp"



#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DTYPE(x, y)                                                 \
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),                            \
                #x " and " #y " must have the same dtype.")

#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

/*
    Torch wrapper for the CUDA kernel forwards pass.
*/
std::vector<torch::Tensor> sphericart_torch::spherical_harmonics_cuda(
    torch::Tensor xyz, torch::Tensor prefactors, int64_t l_max, bool normalize,
    int64_t GRID_DIM_X, int64_t GRID_DIM_Y, bool gradients, bool hessian) {

    CHECK_INPUT(xyz);
    CHECK_INPUT(prefactors);
    CHECK_SAME_DTYPE(xyz, prefactors);

    int n_total = (l_max + 1) * (l_max + 1);

    auto sph = torch::empty(
        {xyz.size(0), n_total},
        torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));

    torch::Tensor d_sph;
    if (xyz.requires_grad() || gradients) {
        d_sph = torch::empty(
            {xyz.size(0), 3, n_total},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
    } else {
        // just so accessor doesn't complain (will be reverted later)
        d_sph = torch::empty(
            {1, 1, 1},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
    }

    torch::Tensor hess_sph;

    if (xyz.requires_grad() && hessian) {
        hess_sph = torch::empty(
            {xyz.size(0), 3, 3, n_total},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
    } else {
        // just so accessor doesn't complain (will be reverted later)
        hess_sph = torch::empty(
            {1, 1, 1, 1},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
    }

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) {
        return (x + bdim - 1) / bdim;
    };

    dim3 block_dim(find_num_blocks(xyz.size(0), GRID_DIM_Y));

    switch (xyz.scalar_type()) {
    case torch::ScalarType::Double:
        spherical_harmonics_cuda_base<double>(
            xyz.data_ptr<double>(), xyz.size(0), prefactors.data_ptr<double>(),
            prefactors.size(0), l_max, normalize, GRID_DIM_X, GRID_DIM_Y,
            xyz.requires_grad(), gradients, hessian, sph.data_ptr<double>(),
            d_sph.data_ptr<double>(), hess_sph.data_ptr<double>());
        break;
    case torch::ScalarType::Float:
        spherical_harmonics_cuda_base<float>(
            xyz.data_ptr<float>(), xyz.size(0), prefactors.data_ptr<float>(),
            prefactors.size(0), l_max, normalize, GRID_DIM_X, GRID_DIM_Y,
            xyz.requires_grad(), gradients, hessian, sph.data_ptr<float>(),
            d_sph.data_ptr<float>(), hess_sph.data_ptr<float>());
        break;
    }

    cudaDeviceSynchronize();

    if (!gradients)
        d_sph = torch::Tensor();
    if (!hessian)
        hess_sph = torch::Tensor();
    return {sph, d_sph, hess_sph};
}

/*
    Torch wrapper for the CUDA kernel backwards pass.
*/
torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad) {

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error(
            "internal error: CUDA version called on non-CUDA tensor");
    }

    auto xyz_grad = torch::Tensor();

    if (xyz.requires_grad()) {
        xyz_grad = torch::empty_like(xyz);

        switch (xyz.scalar_type()) {
        case torch::ScalarType::Double:
            sphericart_torch::spherical_harmonics_backward_cuda_base<double>(
                dsph.data_ptr<double>(), sph_grad.data_ptr<double>(),
                dsph.size(0), sph_grad.size(1), xyz_grad.data_ptr<double>());

            break;
        case torch::ScalarType::Float:
            sphericart_torch::spherical_harmonics_backward_cuda_base<float>(
                dsph.data_ptr<float>(), sph_grad.data_ptr<float>(),
                dsph.size(0), sph_grad.size(1), xyz_grad.data_ptr<float>());
            break;
        }

        cudaDeviceSynchronize();
    }

    return xyz_grad;
}

/*
    wrapper to compute prefactors with correct dtype.

*/

torch::Tensor sphericart_torch::prefactors_cuda(int64_t l_max,
                                                at::ScalarType dtype) {
    auto result =
        torch::empty({(l_max + 1) * (l_max + 2)},
                     torch::TensorOptions().device("cpu").dtype(dtype));

    if (dtype == c10::kDouble) {
        compute_sph_prefactors(l_max, static_cast<double *>(result.data_ptr()));
    } else if (dtype == c10::kFloat) {
        compute_sph_prefactors(l_max, static_cast<float *>(result.data_ptr()));
    } else {
        throw std::runtime_error(
            "this code only runs on float64 and float32 arrays");
    }

    return result.to("cuda");
}
