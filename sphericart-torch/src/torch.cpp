#include <torch/torch.h>

#include "sphericart/torch.hpp"
#include "sphericart/autograd.hpp"

using namespace torch;
using namespace sphericart_torch;

SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool backward_second_derivatives)
    : l_max_(l_max), backward_second_derivatives_(backward_second_derivatives),
      calculator_double_(l_max_), calculator_float_(l_max_) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ptr =
            std::make_unique<sphericart::cuda::SphericalHarmonics<double>>(l_max_);

        this->calculator_cuda_float_ptr =
            std::make_unique<sphericart::cuda::SphericalHarmonics<float>>(l_max_);
    }
}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, true);
}

SolidHarmonics::SolidHarmonics(int64_t l_max, bool backward_second_derivatives)
    : l_max_(l_max), backward_second_derivatives_(backward_second_derivatives),
      calculator_double_(l_max_), calculator_float_(l_max_) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ptr =
            std::make_unique<sphericart::cuda::SolidHarmonics<double>>(l_max_);

        this->calculator_cuda_float_ptr =
            std::make_unique<sphericart::cuda::SolidHarmonics<float>>(l_max_);
    }
}

torch::Tensor SolidHarmonics::compute(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SolidHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SolidHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(
            torch::init<int64_t, bool>(),
            "",
            {torch::arg("l_max"), torch::arg("backward_second_derivatives") = false}
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
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<SphericalHarmonics>& self) -> std::tuple<int64_t, bool> {
                return {self->get_l_max(), self->get_backward_second_derivative_flag()};
            },
            // __setstate__
            [](std::tuple<int64_t, bool> state) -> c10::intrusive_ptr<SphericalHarmonics> {
                const auto l_max = std::get<0>(state);
                const auto backward_second_derivatives = std::get<1>(state);
                return c10::make_intrusive<SphericalHarmonics>(l_max, backward_second_derivatives);
            }
        );

    m.class_<SolidHarmonics>("SolidHarmonics")
        .def(
            torch::init<int64_t, bool>(),
            "",
            {torch::arg("l_max"), torch::arg("backward_second_derivatives") = false}
        )
        .def("compute", &SolidHarmonics::compute, "", {torch::arg("xyz")})
        .def(
            "compute_with_gradients",
            &SolidHarmonics::compute_with_gradients,
            "",
            {torch::arg("xyz")}
        )
        .def(
            "compute_with_hessians",
            &SolidHarmonics::compute_with_hessians,
            "",
            {torch::arg("xyz")}
        )
        .def("omp_num_threads", &SolidHarmonics::get_omp_num_threads)
        .def("l_max", &SolidHarmonics::get_l_max)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<SolidHarmonics>& self) -> std::tuple<int64_t, bool> {
                return {self->get_l_max(), self->get_backward_second_derivative_flag()};
            },
            // __setstate__
            [](std::tuple<int64_t, bool> state) -> c10::intrusive_ptr<SolidHarmonics> {
                const auto l_max = std::get<0>(state);
                const auto backward_second_derivatives = std::get<1>(state);
                return c10::make_intrusive<SolidHarmonics>(l_max, backward_second_derivatives);
            }
        );
}
