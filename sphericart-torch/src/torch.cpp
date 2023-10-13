
#include "sphericart/torch.hpp"

#include <torch/script.h>

#include "sphericart/autograd.hpp"
#include "sphericart/cuda.hpp"

using namespace sphericart_torch;
SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool normalized,
                                       bool backward_second_derivatives)
    : l_max_(l_max), normalized_(normalized),
      backward_second_derivatives_(backward_second_derivatives),
      calculator_double_(l_max_, normalized_),
      calculator_float_(l_max_, normalized_),
      prefactors_cuda_double_(prefactors_cuda(l_max, c10::kDouble)),
      prefactors_cuda_float_(prefactors_cuda(l_max, c10::kFloat)) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();
}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor>
SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor>
SphericalHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(torch::init<int64_t, bool, bool>(), "",
             {torch::arg("l_max"), torch::arg("normalized") = false,
              torch::arg("backward_second_derivatives") = false})
        .def("compute", &SphericalHarmonics::compute, "", {torch::arg("xyz")})
        .def("compute_with_gradients",
             &SphericalHarmonics::compute_with_gradients, "",
             {torch::arg("xyz")})
        .def("compute_with_hessians",
             &SphericalHarmonics::compute_with_hessians, "",
             {torch::arg("xyz")})
        .def("omp_num_threads", &SphericalHarmonics::get_omp_num_threads)
        .def("l_max", &SphericalHarmonics::get_l_max)
        .def("normalized", &SphericalHarmonics::get_normalized_flag);
}
