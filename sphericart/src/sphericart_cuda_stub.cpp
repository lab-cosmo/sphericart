#include <stdexcept>

#include "sphericart_cuda.hpp"

using namespace sphericart::cuda;

template <typename T>
SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {}

template <typename T>
void SphericalHarmonics<T>::compute(const T *xyz, const size_t nsamples,
                                    bool compute_with_gradients,
                                    bool compute_with_hessian, T *sph, T *dsph,
                                    T *ddsph) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}
