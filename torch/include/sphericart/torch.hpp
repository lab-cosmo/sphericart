#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP

#include <mutex>

#include <torch/script.h>

#include "sphericart.hpp"

namespace sphericart_torch {

class SphericalHarmonicsAutograd;

class CudaSharedMemorySettings {
public:
    CudaSharedMemorySettings(): scalar_size_(0), l_max_(-1), grid_dim_y_(-1), requires_grad_(false) {}

    void update_if_required(torch::ScalarType scalar_type, int64_t l_max, int64_t GRID_DIM_Y, bool gradients);
private:
    int64_t l_max_;
    int64_t grid_dim_y_;
    bool requires_grad_;
    size_t scalar_size_;
};

class SphericalHarmonics: public torch::CustomClassHolder {
public:
    SphericalHarmonics(int64_t l_max, bool normalize);

    // Actual calculation, with autograd support
    torch::Tensor compute(torch::Tensor xyz);
    std::vector<torch::Tensor> compute_with_gradients(torch::Tensor xyz);

private:
    friend class SphericalHarmonicsAutograd;

    // Raw calculation, without autograd support, running on CPU
    std::vector<torch::Tensor> compute_raw_cpu(torch::Tensor xyz, bool do_gradients);

    int64_t l_max_;
    bool normalize_;

    // CPU implementation
    sphericart::SphericalHarmonics<double> calculator_double_;
    sphericart::SphericalHarmonics<float> calculator_float_;

    torch::Tensor prefactors_cuda_double_;
    torch::Tensor prefactors_cuda_float_;

    CudaSharedMemorySettings cuda_shmem_;
    std::mutex cuda_shmem_mutex_;
};

} // sphericart_torch
#endif
