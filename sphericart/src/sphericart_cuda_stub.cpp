#include "sphericart_cuda.hpp"

using namespace sphericart::cuda;

template <typename T>
SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {}

template <typename T> void SphericalHarmonics<T>::compute(const T *xyz, const size_t nsamples,
                                    bool compute_with_gradients,
                                    bool compute_with_hessian,
                                    size_t GRID_DIM_X, size_t GRID_DIM_Y,
                                    T *sph, T *dsph = nullptr,
                                    T *ddsph = nullptr) {
    throw std::runtime_error("sphericart was not compiled with CUDA support");
}
