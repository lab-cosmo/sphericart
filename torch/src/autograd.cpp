#include "sphericart/autograd.hpp"
#include "sphericart/torch.hpp"
#include "sphericart.hpp"

using namespace sphericart_torch;
torch::autograd::variable_list SphericalHarmonicsAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    SphericalHarmonics& calculator,
    torch::Tensor xyz
) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cpu()) {
        throw std::runtime_error("this code only runs on CPU for now");
    }

    if (xyz.sizes().size() != 2) {
        throw std::runtime_error("xyz tensor must be a 2D array");
    }

    if (xyz.sizes()[1] != 3) {
        throw std::runtime_error("xyz tensor must be an `n_samples x 3` array");
    }

    auto n_samples = xyz.sizes()[0];
    auto l_max = calculator.l_max;
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph_length = n_samples * (l_max + 1) * (l_max + 1);
    auto dsph_length = n_samples * 3 * (l_max + 1) * (l_max + 1);
    auto sph = torch::zeros({n_samples, (l_max + 1) * (l_max + 1)}, options);

    if (xyz.dtype() == c10::kDouble) {
        auto & sph_calc = calculator.spherical_harmonics_d;

        if (xyz.requires_grad()) {
            auto dsph = torch::zeros({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);

            sph_calc.compute_array(
                xyz.data_ptr<double>(),
                n_samples * 3,
                sph.data_ptr<double>(),
                sph_length,
                dsph.data_ptr<double>(),
                dsph_length
            );
            ctx->save_for_backward({xyz, dsph});
        } else {
            sph_calc.compute_array(
                xyz.data_ptr<double>(),
                n_samples * 3,
                sph.data_ptr<double>(),
                sph_length
            );
        }
    } else if (xyz.dtype() == c10::kFloat) {
        auto & sph_calc = calculator.spherical_harmonics_f;

        if (xyz.requires_grad()) {
            auto dsph = torch::zeros({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);
            sph_calc.compute_array(
                xyz.data_ptr<float>(),
                n_samples * 3,
                sph.data_ptr<float>(),
                sph_length,
                dsph.data_ptr<float>(),
                dsph_length
            );
            ctx->save_for_backward({xyz, dsph});
        } else {
            sph_calc.compute_array(
                xyz.data_ptr<float>(),
                n_samples * 3,
                sph.data_ptr<float>(),
                sph_length
            );
        }
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }

    return {sph};
}

torch::autograd::variable_list SphericalHarmonicsAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    /* get the saved data from the forward pass */
    auto saved_variables = ctx->get_saved_variables();
    auto xyz = saved_variables[0];
    auto dsph = saved_variables[1];

    auto xyz_grad = torch::Tensor();
    if (xyz.requires_grad()) {
        xyz_grad = torch::zeros_like(xyz);

        auto sph_grad = grad_outputs[0];
        if (!sph_grad.device().is_cpu()) {
            throw std::runtime_error("this code only runs on CPU for now");
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
            auto xyz_grad_p = xyz_grad.accessor<double, 2>();
            auto sph_grad_p = sph_grad.accessor<double, 2>();
            auto dsph_p = dsph.accessor<double, 3>();

            for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
                for (size_t spatial=0; spatial<3; spatial++) {
                    for (int i_sph=0; i_sph<n_sph; i_sph++) {
                        xyz_grad_p[i_sample][spatial] += sph_grad_p[i_sample][i_sph] * dsph_p[i_sample][spatial][i_sph];
                    }
                }
            }
        } else if (xyz.dtype() == c10::kFloat) {
            auto xyz_grad_p = xyz_grad.accessor<float, 2>();
            auto sph_grad_p = sph_grad.accessor<float, 2>();
            auto dsph_p = dsph.accessor<float, 3>();

            for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
                for (size_t spatial=0; spatial<3; spatial++) {
                    for (int i_sph=0; i_sph<n_sph; i_sph++) {
                        xyz_grad_p[i_sample][spatial] += sph_grad_p[i_sample][i_sph] * dsph_p[i_sample][spatial][i_sph];
                    }
                }
            }
         } else {
            throw std::runtime_error("this code only runs on float64 and float32 arrays");
        }
    }

    return {torch::Tensor(), xyz_grad, torch::Tensor()};
}
