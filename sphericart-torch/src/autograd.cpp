#include <cstdint> // For intptr_t

#include "sphericart/autograd.hpp"

#include "cuda_base.hpp"
#include "sphericart.hpp"
#include "sphericart/torch.hpp"
#include "sphericart/torch_cuda_wrapper.hpp"
#include <torch/torch.h>

#ifdef CUDA_AVAILABLE
#include <c10/cuda/CUDAStream.h>
#endif

using namespace sphericart_torch;
using namespace at;

template <template <typename> class C, typename scalar_t>
std::vector<torch::Tensor> _compute_raw_cpu(
    C<scalar_t>& calculator, torch::Tensor xyz, int64_t l_max, bool do_gradients, bool do_hessians
) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cpu()) {
        throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
    }

    if (do_hessians && !do_gradients) {
        throw std::runtime_error("internal error: cannot request hessians without gradients");
    }

    auto n_samples = xyz.sizes()[0];
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph_length = n_samples * (l_max + 1) * (l_max + 1);
    auto dsph_length = n_samples * 3 * (l_max + 1) * (l_max + 1);
    auto ddsph_length = n_samples * 9 * (l_max + 1) * (l_max + 1);
    auto sph = torch::empty({n_samples, (l_max + 1) * (l_max + 1)}, options);

    if (do_hessians) {
        auto dsph = torch::empty({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);
        auto ddsph = torch::empty({n_samples, 3, 3, (l_max + 1) * (l_max + 1)}, options);
        calculator.compute_array_with_hessians(
            xyz.data_ptr<scalar_t>(),
            n_samples * 3,
            sph.data_ptr<scalar_t>(),
            sph_length,
            dsph.data_ptr<scalar_t>(),
            dsph_length,
            ddsph.data_ptr<scalar_t>(),
            ddsph_length
        );
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        auto dsph = torch::empty({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);
        calculator.compute_array_with_gradients(
            xyz.data_ptr<scalar_t>(),
            n_samples * 3,
            sph.data_ptr<scalar_t>(),
            sph_length,
            dsph.data_ptr<scalar_t>(),
            dsph_length
        );
        return {sph, dsph, torch::Tensor()};
    } else {
        calculator.compute_array(
            xyz.data_ptr<scalar_t>(), n_samples * 3, sph.data_ptr<scalar_t>(), sph_length
        );
        return {sph, torch::Tensor(), torch::Tensor()};
    }
}

template <template <typename> class C, typename scalar_t>
std::vector<torch::Tensor> _compute_raw_cuda(
    C<scalar_t>* calculator,
    torch::Tensor xyz,
    int64_t l_max,
    bool do_gradients,
    bool do_hessians,
    void* stream
) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error("internal error: called CUDA version on non-CUDA tensor");
    }

    if (do_hessians && !do_gradients) {
        throw std::runtime_error("internal error: cannot request hessians without gradients");
    }

    auto n_samples = xyz.sizes()[0];
    auto lmtotal = (l_max + 1) * (l_max + 1);
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph = torch::empty({n_samples, lmtotal}, options);

    if (do_hessians) {
        auto dsph = torch::empty({n_samples, 3, lmtotal}, options);
        auto ddsph = torch::empty({n_samples, 3, 3, lmtotal}, options);
        calculator->compute_with_hessians(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            dsph.data_ptr<scalar_t>(),
            ddsph.data_ptr<scalar_t>(),
            stream
        );
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        auto dsph = torch::empty({n_samples, 3, lmtotal}, options);
        calculator->compute_with_gradients(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            dsph.data_ptr<scalar_t>(),
            stream
        );
        return {sph, dsph, torch::Tensor()};
    } else {
        calculator->compute(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            reinterpret_cast<void*>(stream)
        );
        return {sph, torch::Tensor(), torch::Tensor()};
    }
}

std::vector<torch::Tensor> SphericalHarmonics::compute_raw_cpu(
    torch::Tensor xyz, bool do_gradients, bool do_hessians
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cpu<sphericart::SphericalHarmonics, double>(
            calculator_double_, xyz, l_max_, do_gradients, do_hessians
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cpu<sphericart::SphericalHarmonics, float>(
            calculator_float_, xyz, l_max_, do_gradients, do_hessians
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

std::vector<torch::Tensor> SphericalHarmonics::compute_raw_cuda(
    torch::Tensor xyz, bool do_gradients, bool do_hessians, void* stream
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cuda<sphericart::cuda::SphericalHarmonics, double>(
            calculator_cuda_double_ptr.get(), xyz, l_max_, do_gradients, do_hessians, stream
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cuda<sphericart::cuda::SphericalHarmonics, float>(
            calculator_cuda_float_ptr.get(), xyz, l_max_, do_gradients, do_hessians, stream
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

std::vector<torch::Tensor> SolidHarmonics::compute_raw_cpu(
    torch::Tensor xyz, bool do_gradients, bool do_hessians
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cpu<sphericart::SolidHarmonics, double>(
            calculator_double_, xyz, l_max_, do_gradients, do_hessians
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cpu<sphericart::SolidHarmonics, float>(
            calculator_float_, xyz, l_max_, do_gradients, do_hessians
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

std::vector<torch::Tensor> SolidHarmonics::compute_raw_cuda(
    torch::Tensor xyz, bool do_gradients, bool do_hessians, void* stream
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cuda<sphericart::cuda::SolidHarmonics, double>(
            calculator_cuda_double_ptr.get(), xyz, l_max_, do_gradients, do_hessians, stream
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cuda<sphericart::cuda::SolidHarmonics, float>(
            calculator_cuda_float_ptr.get(), xyz, l_max_, do_gradients, do_hessians, stream
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

static torch::Tensor backward_cpu(torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad) {
    auto xyz_grad = torch::empty_like(xyz);

    if (!sph_grad.device().is_cpu() || !xyz.device().is_cpu() || !dsph.device().is_cpu()) {
        throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
    }

    // we need contiguous data to take pointers below
    sph_grad = sph_grad.contiguous();

    if (!xyz_grad.is_contiguous() || !dsph.is_contiguous()) {
        // we created these, they should always be contiguous
        throw std::runtime_error("internal error: xyz_grad or dsph are not contiguous");
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
                    accumulated_value += sph_grad_p[n_sph * i_sample + i_sph] *
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
                    accumulated_value += sph_grad_p[n_sph * i_sample + i_sph] *
                                         dsph_p[n_sph * 3 * i_sample + n_sph * spatial + i_sph];
                }
                xyz_grad_p[3 * i_sample + spatial] = accumulated_value;
            }
        }
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }

    return xyz_grad;
}

template <class C>
std::vector<torch::Tensor> SphericartAutograd::forward(
    torch::autograd::AutogradContext* ctx,
    C& calculator,
    torch::Tensor xyz,
    bool do_gradients,
    bool do_hessians
) {
    if (xyz.sizes().size() != 2) {
        throw std::runtime_error("xyz tensor must be a 2D array");
    }

    if (xyz.sizes()[1] != 3) {
        throw std::runtime_error("xyz tensor must be an `n_samples x 3` array");
    }

    void* stream = nullptr;
#ifdef CUDA_AVAILABLE
    stream = reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream());
#endif

    auto sph = torch::Tensor();
    auto dsph = torch::Tensor();
    auto ddsph = torch::Tensor();

    bool requires_grad = do_gradients || xyz.requires_grad();

    bool requires_hessian =
        do_hessians || (xyz.requires_grad() && calculator.backward_second_derivatives_);

    if (xyz.device().is_cpu()) {
        auto results = calculator.compute_raw_cpu(xyz, requires_grad, requires_hessian);
        sph = results[0];
        dsph = results[1];
        ddsph = results[2];
    } else if (xyz.device().is_cuda()) {
        auto results = calculator.compute_raw_cuda(xyz, requires_grad, requires_hessian, stream);
        sph = results[0];
        dsph = results[1];
        ddsph = results[2];

    } else {
        throw std::runtime_error("Spherical harmonics are only implemented for CPU and CUDA");
    }

    if (xyz.requires_grad()) {
        ctx->save_for_backward({xyz, dsph, ddsph});
        ctx->saved_data["stream"] = torch::IValue((int64_t)(intptr_t)stream);
    }

    if (do_hessians) {
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        return {sph, dsph};
    } else {
        return {sph};
    }
}

std::vector<torch::Tensor> SphericartAutograd::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {

    if (grad_outputs.size() > 1) {
        throw std::runtime_error(
            "We can not run a backward pass through the gradients of spherical "
            "harmonics"
        );
    }
    /* get the saved data from the forward pass */
    auto saved_variables = ctx->get_saved_variables();
    // We extract xyz and pass it as a separate variable because we will need
    // gradients with respect to it
    auto xyz = saved_variables[0];
    torch::Tensor xyz_grad =
        SphericartAutogradBackward::apply(grad_outputs[0].contiguous(), xyz, saved_variables);
    return {torch::Tensor(), xyz_grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

torch::Tensor SphericartAutogradBackward::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor grad_outputs,
    torch::Tensor xyz,
    std::vector<torch::Tensor> saved_variables
) {

    void* stream = nullptr;
#ifdef CUDA_AVAILABLE
    stream = reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream());
#endif

    auto dsph = saved_variables[1];
    auto ddsph = saved_variables[2];

    auto xyz_grad = torch::Tensor();
    if (xyz.requires_grad()) {
        if (xyz.device().is_cpu()) {
            xyz_grad = backward_cpu(xyz, dsph, grad_outputs);
        } else if (xyz.device().is_cuda()) {
            xyz_grad =
                sphericart_torch::spherical_harmonics_backward_cuda(xyz, dsph, grad_outputs, stream);
        } else {
            throw std::runtime_error("Spherical harmonics are only implemented for CPU and CUDA");
        }
    }

    ctx->save_for_backward({xyz, grad_outputs, dsph, ddsph});

    return xyz_grad;
}

std::vector<torch::Tensor> SphericartAutogradBackward::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_2_outputs
) {

    auto saved_variables = ctx->get_saved_variables();
    auto xyz = saved_variables[0];
    auto grad_out = saved_variables[1];
    auto dsph = saved_variables[2];
    auto ddsph = saved_variables[3];

    auto grad_2_out = grad_2_outputs[0];

    auto gradgrad_wrt_grad_out = torch::Tensor();
    auto gradgrad_wrt_xyz = torch::Tensor();

    bool double_backward = ddsph.defined(); // If the double backward was not requested in
                                            // advance, this tensor will be uninitialized

    if (!double_backward) {
        TORCH_WARN_ONCE(
            "Second derivatives of the spherical harmonics with respect to the Cartesian "
            "coordinates were not requested at class creation. The second derivative of "
            "the spherical harmonics with respect to the Cartesian coordinates will be "
            "treated as zero, potentially causing incorrect results. Make sure you either "
            "do not need (i.e., are not using) these second derivatives, or that you set "
            "`backward_second_derivatives=True` when creating the SphericalHarmonics or "
            "SolidHarmonics class."
        );
    }

    if (grad_out.requires_grad()) {
        // gradgrad_wrt_grad_out, unlike gradgrad_wrt_xyz, is needed for mixed
        // second derivatives
        int n_samples = xyz.sizes()[0];
        gradgrad_wrt_grad_out = torch::sum(dsph * grad_2_out.reshape({n_samples, 3, 1}), 1);
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
                    torch::sum(grad_out.reshape({n_samples, 1, 1, n_sph}) * ddsph, 3),
                2
            );
            // the above does the same as the following (but faster):
            // gradgrad_wrt_xyz = torch::einsum("sa, sk, sabk -> sb",
            // {grad_2_out, grad_out, ddsph});
            // note that, unlike in the single backward case, we do not provide
            // specific CPU and CUDA kernels for this contraction
        }
        // if double_backward is false, xyz requires a gradient, but the user
        // did not request second derivatives with respect to xyz (and therefore
        // ddsph is an uninitialized tensor). In this case, we return
        // gradgrad_wrt_xyz as an uninitialized tensor: this will signal to
        // PyTorch that the relevant gradients are zero, so that xyz.grad will
        // not be updated
    }

    return {gradgrad_wrt_grad_out, gradgrad_wrt_xyz, torch::Tensor(), torch::Tensor()};
}

// Explicit instantiation of SphericartAutograd::forward
template std::vector<torch::Tensor> SphericartAutograd::forward<SphericalHarmonics>(
    torch::autograd::AutogradContext* ctx,
    SphericalHarmonics& calculator,
    torch::Tensor xyz,
    bool do_gradients,
    bool do_hessians
);
template std::vector<torch::Tensor> SphericartAutograd::forward<SolidHarmonics>(
    torch::autograd::AutogradContext* ctx,
    SolidHarmonics& calculator,
    torch::Tensor xyz,
    bool do_gradients,
    bool do_hessians
);
