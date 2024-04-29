
#include "sphericart/torch.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include "sphericart/autograd.hpp"
#include "sphericart/torch_cuda_wrapper.hpp"

using namespace torch;
using namespace sphericart_torch;

SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool normalized, bool backward_second_derivatives)
    : l_max_(l_max), normalized_(normalized),
      backward_second_derivatives_(backward_second_derivatives),

      calculator_double_(l_max_, normalized_), calculator_float_(l_max_, normalized_)

{
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ =
            sphericart::cuda::SphericalHarmonics<double>(l_max_, normalized_);
        this->calculator_cuda_float_ =
            sphericart::cuda::SphericalHarmonics<float>(l_max_, normalized_);
    }
}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(
            torch::init<int64_t, bool, bool>(),
            "",
            {torch::arg("l_max"),
             torch::arg("normalized") = false,
             torch::arg("backward_second_derivatives") = false}
        )
        .def("compute", &SphericalHarmonics::compute, "", {torch::arg("xyz")})
        .def(
            "compute_with_gradients",
            &SphericalHarmonics::compute_with_gradients,
            "",
            {torch::arg("xyz")}
        )
        .def(
            "compute_with_hessians",
            &SphericalHarmonics::compute_with_hessians,
            "",
            {torch::arg("xyz")}
        )
        .def("omp_num_threads", &SphericalHarmonics::get_omp_num_threads)
        .def("l_max", &SphericalHarmonics::get_l_max)
        .def("normalized", &SphericalHarmonics::get_normalized_flag)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<SphericalHarmonics>& self
            ) -> std::tuple<int64_t, bool, bool> {
                return {
                    self->get_l_max(),
                    self->get_normalized_flag(),
                    self->get_backward_second_derivative_flag()
                };
            },
            // __setstate__
            [](std::tuple<int64_t, bool, bool> state) -> c10::intrusive_ptr<SphericalHarmonics> {
                const auto l_max = std::get<0>(state);
                const auto normalized = std::get<1>(state);
                const auto backward_second_derivatives = std::get<2>(state);
                return c10::make_intrusive<SphericalHarmonics>(
                    l_max, normalized, backward_second_derivatives
                );
            }
        );
}
