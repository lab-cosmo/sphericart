#include <torch/script.h>

#include "sphericart/torch.hpp"
#include "sphericart/autograd.hpp"

torch::Tensor sphericart::spherical_harmonics(int64_t l_max, torch::Tensor xyz, bool normalize) {
    auto result = sphericart::SphericalHarmonicsAutograd::apply(l_max, xyz, normalize);

    return result[0];
}

//============================================================================//

TORCH_LIBRARY(sphericart, m) {
    m.def(
        "spherical_harmonics(int l_max, Tensor xyz, bool normalize) -> Tensor",
        sphericart::spherical_harmonics
    );
}
