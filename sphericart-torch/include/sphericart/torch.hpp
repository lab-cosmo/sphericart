#ifndef SPHERICART_TORCH_HPP
#define SPHERICART_TORCH_HPP

#include <torch/torch.h>

#include "sphericart.hpp"
#include "sphericart_cuda.hpp"

namespace sphericart_torch {

template <template <typename> class CPUCalculator, template <typename> class CUDACalculator>
class Harmonics {
    // a class that wraps both CPU and CUDA implementations of either spherical or solid harmonics
  public:
    explicit Harmonics(int64_t l_max);

    int64_t get_omp_num_threads() const { return this->omp_num_threads_; }

    // Raw calculation, without autograd support, running on CPU
    std::vector<torch::Tensor> compute_raw_cpu(torch::Tensor xyz, bool do_gradients, bool do_hessians);
    // Raw calculation, without autograd support, running on CUDA
    std::vector<torch::Tensor> compute_raw_cuda(
        torch::Tensor xyz, bool do_gradients, bool do_hessians, void* stream = nullptr
    );

    int64_t omp_num_threads_;
    int64_t l_max_;

    // CPU implementation
    CPUCalculator<double> calculator_double_;
    CPUCalculator<float> calculator_float_;

    // CUDA implementation
    std::unique_ptr<CUDACalculator<double>> calculator_cuda_double_ptr;
    std::unique_ptr<CUDACalculator<float>> calculator_cuda_float_ptr;
};

using SphericalHarmonics =
    Harmonics<sphericart::SphericalHarmonics, sphericart::cuda::SphericalHarmonics>;
using SolidHarmonics = Harmonics<sphericart::SolidHarmonics, sphericart::cuda::SolidHarmonics>;

} // namespace sphericart_torch
#endif
