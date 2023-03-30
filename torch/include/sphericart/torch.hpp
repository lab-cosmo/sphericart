#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP
#include "sphericart.hpp"
#include <torch/script.h>

namespace sphericart_torch {

class SphericalHarmonicsAutograd;

class SphericalHarmonics: public torch::CustomClassHolder {
public:
    SphericalHarmonics(int64_t l_max, bool normalized);

    torch::Tensor compute(torch::Tensor xyz);


private:
    friend class SphericalHarmonicsAutograd;

    int64_t l_max;
    sphericart::SphericalHarmonics<double> spherical_harmonics_d;
    sphericart::SphericalHarmonics<float> spherical_harmonics_f;
};

} // sphericart_torch
#endif
