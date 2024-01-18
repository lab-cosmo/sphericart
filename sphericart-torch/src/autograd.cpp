#ifdef CUDA_AVAILABLE
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <iostream>

#include "sphericart/autograd.hpp"

#include "cuda_base.hpp"
#include "sphericart.hpp"
#include "sphericart/torch.hpp"
#include "sphericart/torch_cuda_wrapper.hpp"
#include <torch/torch.h>

using namespace std;
using namespace sphericart_torch;

std::vector<torch::Tensor>
SphericalHarmonics::compute_raw_cpu(torch::Tensor xyz, bool do_gradients,
                                    bool do_hessians) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cpu()) {
        throw std::runtime_error(
            "internal error: called CPU version on non-CPU tensor");
    }

    if (do_hessians && !do_gradients) {
        throw std::runtime_error(
            "internal error: cannot request hessians without gradients");
    }

    auto n_samples = xyz.sizes()[0];
    auto options =
        torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph_length = n_samples * (l_max_ + 1) * (l_max_ + 1);
    auto dsph_length = n_samples * 3 * (l_max_ + 1) * (l_max_ + 1);
    auto ddsph_length = n_samples * 9 * (l_max_ + 1) * (l_max_ + 1);
    auto sph = torch::empty({n_samples, (l_max_ + 1) * (l_max_ + 1)}, options);

    if (xyz.dtype() == c10::kDouble) {
        if (do_hessians) {
            auto dsph = torch::empty(
                {n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            auto ddsph = torch::empty(
                {n_samples, 3, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            calculator_double_.compute_array_with_hessians(
                xyz.data_ptr<double>(), n_samples * 3, sph.data_ptr<double>(),
                sph_length, dsph.data_ptr<double>(), dsph_length,
                ddsph.data_ptr<double>(), ddsph_length);
            return {sph, dsph, ddsph};
        } else if (do_gradients) {
            auto dsph = torch::empty(
                {n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            calculator_double_.compute_array_with_gradients(
                xyz.data_ptr<double>(), n_samples * 3, sph.data_ptr<double>(),
                sph_length, dsph.data_ptr<double>(), dsph_length);
            return {sph, dsph, torch::Tensor()};
        } else {
            calculator_double_.compute_array(
                xyz.data_ptr<double>(), n_samples * 3, sph.data_ptr<double>(),
                sph_length);
            return {sph, torch::Tensor(), torch::Tensor()};
        }
    } else if (xyz.dtype() == c10::kFloat) {
        if (do_hessians) {
            auto dsph = torch::empty(
                {n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            auto ddsph = torch::empty(
                {n_samples, 3, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            calculator_float_.compute_array_with_hessians(
                xyz.data_ptr<float>(), n_samples * 3, sph.data_ptr<float>(),
                sph_length, dsph.data_ptr<float>(), dsph_length,
                ddsph.data_ptr<float>(), ddsph_length);
            return {sph, dsph, ddsph};
        } else if (do_gradients) {
            auto dsph = torch::empty(
                {n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            calculator_float_.compute_array_with_gradients(
                xyz.data_ptr<float>(), n_samples * 3, sph.data_ptr<float>(),
                sph_length, dsph.data_ptr<float>(), dsph_length);
            return {sph, dsph, torch::Tensor()};
        } else {
            calculator_float_.compute_array(xyz.data_ptr<float>(),
                                            n_samples * 3,
                                            sph.data_ptr<float>(), sph_length);
            return {sph, torch::Tensor(), torch::Tensor()};
        }
    } else {
        throw std::runtime_error(
            "this code only runs on float64 and float32 arrays");
    }
}

static torch::Tensor backward_cpu(torch::Tensor xyz, torch::Tensor dsph,
                                  torch::Tensor sph_grad) {
    auto xyz_grad = torch::empty_like(xyz);

    if (!sph_grad.device().is_cpu() || !xyz.device().is_cpu() ||
        !dsph.device().is_cpu()) {
        throw std::runtime_error(
            "internal error: called CPU version on non-CPU tensor");
    }

    // we need contiguous data to take pointers below
    sph_grad = sph_grad.contiguous();

    if (!xyz_grad.is_contiguous() || !dsph.is_contiguous()) {
        // we created these, they should always be contiguous
        throw std::runtime_error(
            "internal error: xyz_grad or dsph are not contiguous");
    }

    auto n_samples = xyz.sizes()[0];
    auto n_sph = sph_grad.sizes()[1];
    if (xyz.dtype() == c10::kDouble) {
        auto xyz_grad_p = xyz_grad.data_ptr<double>();
        auto sph_grad_p = sph_grad.data_ptr<double>();
        auto dsph_p = dsph.data_ptr<double>();

#pragma omp parallel for
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            for (size_t spatial = 0; spatial < 3; spatial++) {
                double accumulated_value = 0.0;
                for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                    accumulated_value +=
                        sph_grad_p[n_sph * i_sample + i_sph] *
                        dsph_p[n_sph * 3 * i_sample + n_sph * spatial + i_sph];
                }
                xyz_grad_p[3 * i_sample + spatial] = accumulated_value;
            }
        }
    } else if (xyz.dtype() == c10::kFloat) {
        auto xyz_grad_p = xyz_grad.data_ptr<float>();
        auto sph_grad_p = sph_grad.data_ptr<float>();
        auto dsph_p = dsph.data_ptr<float>();

#pragma omp parallel for
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            for (size_t spatial = 0; spatial < 3; spatial++) {
                float accumulated_value = 0.0f;
                for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                    accumulated_value +=
                        sph_grad_p[n_sph * i_sample + i_sph] *
                        dsph_p[n_sph * 3 * i_sample + n_sph * spatial + i_sph];
                }
                xyz_grad_p[3 * i_sample + spatial] = accumulated_value;
            }
        }
    } else {
        throw std::runtime_error(
            "this code only runs on float64 and float32 arrays");
    }

    return xyz_grad;
}

torch::autograd::variable_list SphericalHarmonicsAutograd::forward(
    torch::autograd::AutogradContext *ctx, SphericalHarmonics &calculator,
    torch::Tensor xyz, bool do_gradients, bool do_hessians) {
    if (xyz.sizes().size() != 2) {
        throw std::runtime_error("xyz tensor must be a 2D array");
    }

    if (xyz.sizes()[1] != 3) {
        throw std::runtime_error("xyz tensor must be an `n_samples x 3` array");
    }

    auto sph = torch::Tensor();
    auto dsph = torch::Tensor();
    auto ddsph = torch::Tensor();

    bool requires_grad = do_gradients || xyz.requires_grad();

    bool requires_hessian =
        do_hessians ||
        (xyz.requires_grad() && calculator.backward_second_derivatives_);

    if (xyz.device().is_cpu()) {
        auto results =
            calculator.compute_raw_cpu(xyz, requires_grad, requires_hessian);
        sph = results[0];
        dsph = results[1];
        ddsph = results[2];
    } else if (xyz.device().is_cuda()) {

        void *stream = nullptr;
#ifdef CUDA_AVAILABLE
        c10::cuda::CUDAGuard deviceGuard{xyz.device()};
        cudaStream_t currstream = c10::cuda::getCurrentCUDAStream();
        stream = reinterpret_cast<void *>(currstream);
#endif

        int n_total = (calculator.l_max_ + 1) * (calculator.l_max_ + 1);

        sph = torch::empty(
            {xyz.size(0), n_total},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));

        if (requires_grad) {
            dsph = torch::empty(
                {xyz.size(0), 3, n_total},
                torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
        } else {
            // just so accessor doesn't complain (will be reverted later)
            dsph = torch::empty(
                {1, 1, 1},
                torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
        }

        if (requires_hessian) {
            ddsph = torch::empty(
                {xyz.size(0), 3, 3, n_total},
                torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
        } else {
            // just so accessor doesn't complain (will be reverted later)
            ddsph = torch::empty(
                {1, 1, 1, 1},
                torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
        }

        if (xyz.dtype() == c10::kDouble) {
            calculator.calculator_cuda_double_.compute(
                xyz.data_ptr<double>(), xyz.size(0), requires_grad,
                requires_hessian, sph.data_ptr<double>(),
                dsph.data_ptr<double>(), ddsph.data_ptr<double>(), stream);

        } else if (xyz.dtype() == c10::kFloat) {
            calculator.calculator_cuda_float_.compute(
                xyz.data_ptr<float>(), xyz.size(0), requires_grad,
                requires_hessian, sph.data_ptr<float>(), dsph.data_ptr<float>(),
                ddsph.data_ptr<float>(), stream);
        } else {
            throw std::runtime_error(
                "this code only runs on c10::kDouble and c10::kFloat tensors");
        }

        if (!requires_grad) {
            dsph = torch::Tensor();
        }

        if (!requires_hessian) {
            ddsph = torch::Tensor();
        }

    } else {
        throw std::runtime_error(
            "Spherical harmonics are only implemented for CPU and CUDA");
    }

    if (xyz.requires_grad()) {
        ctx->save_for_backward({xyz, dsph, ddsph});
    }

    if (do_hessians) {
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        return {sph, dsph};
    } else {
        return {sph};
    }
}

torch::autograd::variable_list SphericalHarmonicsAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {
    if (grad_outputs.size() > 1) {
        throw std::runtime_error(
            "We can not run a backward pass through the gradients of spherical "
            "harmonics");
    }

    /* get the saved data from the forward pass */
    auto saved_variables = ctx->get_saved_variables();
    // We extract xyz and pass it as a separate variable because we will need
    // gradients with respect to it
    auto xyz = saved_variables[0];
    torch::Tensor xyz_grad = SphericalHarmonicsAutogradBackward::apply(
        grad_outputs[0], xyz, saved_variables);
    return {torch::Tensor(), xyz_grad, torch::Tensor(), torch::Tensor()};
}

torch::Tensor SphericalHarmonicsAutogradBackward::forward(
    torch::autograd::AutogradContext *ctx, torch::Tensor grad_outputs,
    torch::Tensor xyz, std::vector<torch::Tensor> saved_variables) {
    auto dsph = saved_variables[1];
    auto ddsph = saved_variables[2];

    auto xyz_grad = torch::Tensor();
    if (xyz.requires_grad()) {
        if (xyz.device().is_cpu()) {
            xyz_grad = backward_cpu(xyz, dsph, grad_outputs);
        } else if (xyz.device().is_cuda()) {
            void *stream = nullptr;
#ifdef CUDA_AVAILABLE
            c10::cuda::CUDAGuard deviceGuard{xyz.device()};
            cudaStream_t currstream = c10::cuda::getCurrentCUDAStream();
            stream = reinterpret_cast<void *>(currstream);
#endif
            xyz_grad = sphericart_torch::spherical_harmonics_backward_cuda(
                xyz, dsph, grad_outputs, stream);
        } else {
            throw std::runtime_error(
                "Spherical harmonics are only implemented for CPU and CUDA");
        }
    }

    ctx->save_for_backward({xyz, grad_outputs, dsph, ddsph});

    return xyz_grad;
}

torch::autograd::variable_list SphericalHarmonicsAutogradBackward::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_2_outputs) {
    auto saved_variables = ctx->get_saved_variables();
    auto xyz = saved_variables[0];
    auto grad_out = saved_variables[1];
    auto dsph = saved_variables[2];
    auto ddsph = saved_variables[3];

    auto grad_2_out = grad_2_outputs[0];

    auto gradgrad_wrt_grad_out = torch::Tensor();
    auto gradgrad_wrt_xyz = torch::Tensor();

    bool double_backward =
        ddsph.defined(); // If the double backward was not requested in
                         // advance, this  tensor will be uninitialized

    if (grad_out.requires_grad()) {
        // gradgrad_wrt_grad_out, unlike gradgrad_wrt_xyz, is needed for mixed
        // second derivatives
        int n_samples = xyz.sizes()[0];
        gradgrad_wrt_grad_out =
            torch::sum(dsph * grad_2_out.reshape({n_samples, 3, 1}), 1);
        // the above does the same as the following (but faster):
        // gradgrad_wrt_grad_out = torch::einsum("sak, sa -> sk", {dsph,
        // grad_2_out});
    }

    if (xyz.requires_grad()) {
        // gradgrad_wrt_xyz is needed to differentiate twice with respect to
        // xyz. However, we only do this if the user requested it when creating
        // the class
        if (double_backward) {
            int n_samples = xyz.size(0);
            int n_sph = grad_out.size(1);
            gradgrad_wrt_xyz = torch::sum(
                grad_2_out.reshape({n_samples, 1, 3}) *
                    torch::sum(
                        grad_out.reshape({n_samples, 1, 1, n_sph}) * ddsph, 3),
                2);
            // the above does the same as the following (but faster):
            // gradgrad_wrt_xyz = torch::einsum("sa, sk, sabk -> sb",
            // {grad_2_out, grad_out, ddsph});
        }
        // if double_backward is false, xyz requires a gradient, but the user
        // did not request second derivatives with respect to xyz (and therefore
        // ddsph is an uninitialized tensor). In this case, we return
        // gradgrad_wrt_xyz as an uninitialized tensor: this will signal to
        // PyTorch that the relevant gradients are zero, so that xyz.grad will
        // not be updated
    }

    return {gradgrad_wrt_grad_out, gradgrad_wrt_xyz, torch::Tensor()};
}
