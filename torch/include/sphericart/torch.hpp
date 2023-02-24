#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP

#include <ATen/Tensor.h>

namespace sphericart {

at::Tensor spherical_harmonics(int64_t l_max, at::Tensor xyz, bool normalize);

}

#endif
