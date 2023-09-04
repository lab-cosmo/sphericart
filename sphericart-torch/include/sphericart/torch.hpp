#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP

#include <torch/script.h>

#include <mutex>

#include "sphericart.hpp"

namespace sphericart_torch {

class SphericalHarmonicsAutograd;
class SphericalHarmonicsAutogradBackward;

class CudaSharedMemorySettings {
  public:
    CudaSharedMemorySettings()
        : scalar_size_(0), l_max_(-1), grid_dim_x_(-1), grid_dim_y_(-1),
          requires_grad_(false), requires_hessian_(false) {}

    bool update_if_required(torch::ScalarType scalar_type, int64_t l_max,
                            int64_t GRID_DIM_X, int64_t GRID_DIM_Y,
                            bool gradients, bool hessian);

  private:
    int64_t l_max_;
    int64_t grid_dim_x_;
    int64_t grid_dim_y_;
    bool requires_grad_;
    bool requires_hessian_;
    size_t scalar_size_;
};

class SphericalHarmonics : public torch::CustomClassHolder {
  public:
    SphericalHarmonics(int64_t l_max, bool normalized = false,
                       bool backward_second_derivatives = false);

    // Actual calculation, with autograd support
    torch::Tensor compute(torch::Tensor xyz);
    std::vector<torch::Tensor> compute_with_gradients(torch::Tensor xyz);
    std::vector<torch::Tensor> compute_with_hessians(torch::Tensor xyz);

    int64_t get_l_max() const { return this->l_max_; }
    bool get_backward_second_derivative_flag() const {
        return this->backward_second_derivatives_;
    }
    bool get_normalized_flag() const { return this->normalized_; }
    int64_t get_omp_num_threads() const { return this->omp_num_threads_; }

  private:
    friend class SphericalHarmonicsAutograd;

    // Raw calculation, without autograd support, running on CPU
    std::vector<torch::Tensor>
    compute_raw_cpu(torch::Tensor xyz, bool do_gradients, bool do_hessians);

    int64_t omp_num_threads_;
    int64_t l_max_;
    bool backward_second_derivatives_;
    bool normalized_;

    // CPU implementation
    sphericart::SphericalHarmonics<double> calculator_double_;
    sphericart::SphericalHarmonics<float> calculator_float_;

    // CUDA sdata
    torch::Tensor prefactors_cuda_double_;
    torch::Tensor prefactors_cuda_float_;

    int64_t CUDA_GRID_DIM_X_ = 8;
    int64_t CUDA_GRID_DIM_Y_ = 8;
    CudaSharedMemorySettings cuda_shmem_;
    std::mutex cuda_shmem_mutex_;
};

} // namespace sphericart_torch
#endif
