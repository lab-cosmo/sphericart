#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP
#include "sphericart.hpp"
#include <torch/script.h>

namespace sphericart_torch {
    class SphericalHarmonicsAutograd;
    class SphericalHarmonics : public torch::CustomClassHolder {
    public:
        SphericalHarmonics(int64_t l_max, bool normalized);

        torch::Tensor compute(torch::Tensor xyz);

    friend class SphericalHarmonicsAutograd;
    private:
        int64_t l_max;
        sphericart::SphericalHarmonics<double> spherical_harmonics;
    };
} // sphericart_torch
#endif
