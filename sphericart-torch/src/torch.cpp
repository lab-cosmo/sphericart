#include <torch/torch.h>

#include "sphericart/torch.hpp"
#include "sphericart/autograd.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

using namespace sphericart_torch;

namespace {

template <typename Calculator>
using CacheMap = std::map<std::tuple<int64_t, bool>, std::unique_ptr<Calculator>>;

template <typename Calculator>
Calculator& get_or_create_calculator(int64_t l_max, bool backward_second_derivatives) {
    static CacheMap<Calculator> cache;
    static std::mutex cache_mutex;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto key = std::make_tuple(l_max, backward_second_derivatives);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it =
            cache.insert({key, std::make_unique<Calculator>(l_max, backward_second_derivatives)}).first;
    }
    return *(it->second);
}

template <typename Calculator>
torch::Tensor compute(torch::Tensor xyz, int64_t l_max, bool backward_second_derivatives) {
    auto& calculator = get_or_create_calculator<Calculator>(l_max, backward_second_derivatives);
    return calculator.compute(xyz);
}

template <typename Calculator>
std::tuple<torch::Tensor, torch::Tensor> compute_with_gradients(torch::Tensor xyz, int64_t l_max) {
    auto& calculator = get_or_create_calculator<Calculator>(l_max, false);
    auto result = calculator.compute_with_gradients(xyz);
    return {result[0], result[1]};
}

template <typename Calculator>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_with_hessians(
    torch::Tensor xyz, int64_t l_max
) {
    auto& calculator = get_or_create_calculator<Calculator>(l_max, false);
    auto result = calculator.compute_with_hessians(xyz);
    return {result[0], result[1], result[2]};
}

template <typename Calculator> int64_t omp_num_threads(int64_t l_max) {
    auto& calculator = get_or_create_calculator<Calculator>(l_max, false);
    return calculator.get_omp_num_threads();
}

} // namespace

SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool backward_second_derivatives)
    : l_max_(l_max), backward_second_derivatives_(backward_second_derivatives),
      calculator_double_(l_max_), calculator_float_(l_max_) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ptr =
            std::make_unique<sphericart::cuda::SphericalHarmonics<double>>(l_max_);

        this->calculator_cuda_float_ptr =
            std::make_unique<sphericart::cuda::SphericalHarmonics<float>>(l_max_);
    }
}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, true);
}

SolidHarmonics::SolidHarmonics(int64_t l_max, bool backward_second_derivatives)
    : l_max_(l_max), backward_second_derivatives_(backward_second_derivatives),
      calculator_double_(l_max_), calculator_float_(l_max_) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ptr =
            std::make_unique<sphericart::cuda::SolidHarmonics<double>>(l_max_);

        this->calculator_cuda_float_ptr =
            std::make_unique<sphericart::cuda::SolidHarmonics<float>>(l_max_);
    }
}

torch::Tensor SolidHarmonics::compute(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SolidHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SolidHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericartAutograd::apply(*this, xyz, true, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.def("spherical_harmonics(Tensor xyz, int l_max, bool backward_second_derivatives=False) -> Tensor");
    m.def("spherical_harmonics_with_gradients(Tensor xyz, int l_max) -> (Tensor, Tensor)");
    m.def("spherical_harmonics_with_hessians(Tensor xyz, int l_max) -> (Tensor, Tensor, Tensor)");
    m.def("solid_harmonics(Tensor xyz, int l_max, bool backward_second_derivatives=False) -> Tensor");
    m.def("solid_harmonics_with_gradients(Tensor xyz, int l_max) -> (Tensor, Tensor)");
    m.def("solid_harmonics_with_hessians(Tensor xyz, int l_max) -> (Tensor, Tensor, Tensor)");
    m.def("spherical_harmonics_omp_num_threads(int l_max) -> int");
    m.def("solid_harmonics_omp_num_threads(int l_max) -> int");
}

TORCH_LIBRARY_IMPL(sphericart_torch, CPU, m) {
    m.impl("spherical_harmonics", &compute<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_gradients", &compute_with_gradients<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_hessians", &compute_with_hessians<SphericalHarmonics>);
    m.impl("solid_harmonics", &compute<SolidHarmonics>);
    m.impl("solid_harmonics_with_gradients", &compute_with_gradients<SolidHarmonics>);
    m.impl("solid_harmonics_with_hessians", &compute_with_hessians<SolidHarmonics>);
}

TORCH_LIBRARY_IMPL(sphericart_torch, CUDA, m) {
    m.impl("spherical_harmonics", &compute<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_gradients", &compute_with_gradients<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_hessians", &compute_with_hessians<SphericalHarmonics>);
    m.impl("solid_harmonics", &compute<SolidHarmonics>);
    m.impl("solid_harmonics_with_gradients", &compute_with_gradients<SolidHarmonics>);
    m.impl("solid_harmonics_with_hessians", &compute_with_hessians<SolidHarmonics>);
}

TORCH_LIBRARY_IMPL(sphericart_torch, CompositeExplicitAutograd, m) {
    m.impl("spherical_harmonics_omp_num_threads", &omp_num_threads<SphericalHarmonics>);
    m.impl("solid_harmonics_omp_num_threads", &omp_num_threads<SolidHarmonics>);
}
