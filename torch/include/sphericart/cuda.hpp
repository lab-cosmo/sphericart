#ifndef SPHERICART_TORCH_CUDA_HPP
#define SPHERICART_TORCH_CUDA_HPP

#include <ATen/Tensor.h>
#include <vector>

namespace sphericart_torch {

bool adjust_cuda_shared_memory(at::ScalarType scalar_type, int64_t l_max, int64_t GRID_DIM_X, int64_t GRID_DIM_Y, bool requires_grad);

std::vector<at::Tensor> spherical_harmonics_cuda(
    at::Tensor xyz,
    at::Tensor prefactors,
    int64_t l_max,
    bool normalize,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool gradients
);

at::Tensor spherical_harmonics_backward_cuda(at::Tensor xyz, at::Tensor dsph, at::Tensor sph_grad);

at::Tensor prefactors_cuda(int64_t l_max, at::ScalarType dtype);

}

#endif
