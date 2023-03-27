
#include <torch/script.h>

#include "sphericart/torch.hpp"
#include "sphericart/autograd.hpp"
#include "sphericart/cuda.hpp"

using namespace sphericart_torch;
SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool normalize):
    l_max_(l_max),
    normalize_(normalize),
    calculator_double_(l_max, normalize),
    calculator_float_(l_max, normalize),
    prefactors_cuda_double_(prefactors_cuda(l_max, c10::kDouble)),
    prefactors_cuda_float_(prefactors_cuda(l_max, c10::kFloat))
{}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, false)[0];
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(torch::init<int64_t, bool>())
        .def("compute", &SphericalHarmonics::compute)
        .def("compute_with_gradients", &SphericalHarmonics::compute_with_gradients);
}
