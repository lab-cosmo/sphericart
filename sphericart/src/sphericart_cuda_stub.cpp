#include <stdexcept>

#include "sphericart_cuda.hpp"

using namespace sphericart::cuda;

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t /*l_max*/) {}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {}

template <typename T>
void SphericalHarmonics<T>::compute_internal(
    T* /*xyz*/,
    const size_t /*n_samples*/,
    bool /*compute_with_gradients*/,
    bool /*compute_with_hessian*/,
    T* /*sph*/,
    T* /*dsph*/,
    T* /*ddsph*/,
    void* /*cuda_stream*/
) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template <typename T>
void SphericalHarmonics<T>::compute(
    T* /*xyz*/, const size_t /*n_samples*/, T* /*sph*/, void* /*cuda_stream*/
) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    T* /*xyz*/, const size_t /*n_samples*/, T* /*sph*/, T* /*dsph*/, void* /*cuda_stream*/
) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    T* /*xyz*/, const size_t /*n_samples*/, T* /*sph*/, T* /*dsph*/, T* /*ddsph*/, void* /*cuda_stream*/
) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}

template <typename T>
SolidHarmonics<T>::SolidHarmonics(size_t l_max) : SphericalHarmonics<T>(l_max) {}

// instantiates the SphericalHarmonics and SolidHarmonics classes
// for basic floating point types
template class sphericart::cuda::SphericalHarmonics<float>;
template class sphericart::cuda::SphericalHarmonics<double>;
template class sphericart::cuda::SolidHarmonics<float>;
template class sphericart::cuda::SolidHarmonics<double>;
