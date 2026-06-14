#include <iostream>
#include <stdexcept>

#include "sycl_device.hpp"

#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include "sphericart.hpp"
#include "sphericart_sycl.hpp"
#include "templates_core.hpp"
#include "sycl_base.hpp"

namespace sphericart {
namespace sycl {

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t l_max) {
    /*
     * Initializes buffer space, computes prefactors, and sets the function
     * pointers used for the actual calls.
     */
    this->l_max = (int)l_max;
    this->nprefactors = (int)(l_max + 1) * (l_max + 2);
    this->normalized = true; // SphericalHarmonics class
    this->prefactors_cpu = new T[this->nprefactors];

    // compute prefactors on host
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    if (this->prefactors_cpu != nullptr) {
        delete[] (this->prefactors_cpu);
        this->prefactors_cpu = nullptr;
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_internal(
    const T* xyz,
    const size_t n_samples,
    bool compute_with_gradients,
    bool compute_with_hessian,
    T* sph,
    T* dsph,
    T* ddsph
) {
    sphericart::sycl::spherical_harmonics_sycl_base<T>(
        xyz,
        n_samples,
        this->prefactors_cpu,
        this->nprefactors,
        this->l_max,
        this->normalized,
        compute_with_gradients,
        compute_with_hessian,
        sph,
        dsph,
        ddsph
    );
}
template <typename T>
void SphericalHarmonics<T>::compute(const T* xyz, const size_t n_samples, T* sph) {

    std::vector<T> ddsph;
    std::vector<T> dsph;
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, false, false, sph, dsph.data(), ddsph.data()
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    const T* xyz, const size_t n_samples, T* sph, T* dsph
) {

    std::vector<T> ddsph;

    SphericalHarmonics<T>::compute_internal(xyz, n_samples, true, false, sph, dsph, ddsph.data());
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, T* ddsph
) {
    SphericalHarmonics<T>::compute_internal(xyz, n_samples, true, true, sph, dsph, ddsph);
}

template <typename T>
SolidHarmonics<T>::SolidHarmonics(size_t l_max) : SphericalHarmonics<T>(l_max) {
    this->normalized = false; // SolidHarmonics class
}

// instantiates the SphericalHarmonics and SolidHarmonics classes
// for basic floating point types
template class sphericart::sycl::SphericalHarmonics<float>;
template class sphericart::sycl::SphericalHarmonics<double>;
template class sphericart::sycl::SolidHarmonics<float>;
template class sphericart::sycl::SolidHarmonics<double>;

} // namespace sycl
} // namespace sphericart
