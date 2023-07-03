#include "sphericart/cuda.hpp"

#include <stdexcept>

bool sphericart_torch::adjust_cuda_shared_memory(at::ScalarType, int64_t, int64_t, int64_t, bool, bool) {
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

std::vector<at::Tensor> sphericart_torch::spherical_harmonics_cuda(at::Tensor, at::Tensor, int64_t, bool, int64_t, int64_t, bool, bool) {
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

at::Tensor sphericart_torch::spherical_harmonics_backward_cuda(at::Tensor, at::Tensor, at::Tensor) {
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

at::Tensor sphericart_torch::prefactors_cuda(int64_t, at::ScalarType) {
    return at::Tensor();
}
