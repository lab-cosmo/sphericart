#include <torch/script.h>
#include "sphericart/torch.hpp"
#include "sphericart/autograd.hpp"

using namespace sphericart_torch;
SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool normalized) : l_max(l_max), spherical_harmonics(l_max, normalized) {}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz)[0];
}
//============================================================================//

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(torch::init<int64_t, bool>())
        .def("compute", &SphericalHarmonics::compute);
}