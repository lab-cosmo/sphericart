#include <cmath>
#include <iostream>
#include "sphericart.hpp"
#include "templates.hpp"

// macro to define different possible hardcoded function calls
#define _HARCODED_SWITCH_CASE(L_MAX) \
    if (this->normalized) { \
        this->_array_no_derivatives = &hardcoded_sph<T, false, true, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<T, true, true, L_MAX>; \
        this->_sample_no_derivatives = &hardcoded_sph_sample<T, false, true, SPHERICART_LMAX_HARDCODED>; \
        this->_sample_with_derivatives = &hardcoded_sph_sample<T, true, true, SPHERICART_LMAX_HARDCODED>; \
    } else { \
        this->_array_no_derivatives = &hardcoded_sph<T, false, false, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<T, true, false, L_MAX>; \
        this->_sample_no_derivatives = &hardcoded_sph_sample<T, false, false, SPHERICART_LMAX_HARDCODED>; \
        this->_sample_with_derivatives = &hardcoded_sph_sample<T, true, false, SPHERICART_LMAX_HARDCODED>; \
    }

template<typename T>
sphericart::SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {
    this->l_max = (int) l_max;
    this->size_y = (int) (l_max + 1) * (l_max + 1);
    this->size_q = (int) (l_max + 1) * (l_max + 2) / 2;
    this->normalized = normalized;
    this->prefactors = new T[(l_max+1)*(l_max+2)];

    // buffers for cos, sin, 2mz arrays
#ifdef _OPENMP
    // allocates buffers that are large enough to store thread-local data
    this->buffers = new T[this->size_q*3*omp_get_max_threads()];
#else
    this->buffers = new T[this->size_q*3];
#endif
    compute_sph_prefactors<T>((int) l_max, this->prefactors);

    // sets the correct function pointers for the compute functions
    if (this->l_max<=SPHERICART_LMAX_HARDCODED) {
        switch (this->l_max) {
        case 0:
            _HARCODED_SWITCH_CASE(0);
            break;
        case 1:
            _HARCODED_SWITCH_CASE(1);
            break;
        case 2:
            _HARCODED_SWITCH_CASE(2);
            break;
        case 3:
            _HARCODED_SWITCH_CASE(3);
            break;
        case 4:
            _HARCODED_SWITCH_CASE(4);
            break;
        case 5:
            _HARCODED_SWITCH_CASE(5);
            break;
        case 6:
            _HARCODED_SWITCH_CASE(6);
            break;
        }
    } else {
        if (this->normalized) {
            this->_array_no_derivatives = &generic_sph<T, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<T, true, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives = &generic_sph_sample<T, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives = &generic_sph_sample<T, true, true, SPHERICART_LMAX_HARDCODED>;
        } else {
            this->_array_no_derivatives = &generic_sph<T, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<T, true, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives = &generic_sph_sample<T, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives = &generic_sph_sample<T, true, false, SPHERICART_LMAX_HARDCODED>;
        }
    }
}

template<typename T>
sphericart::SphericalHarmonics<T>::~SphericalHarmonics() {
    delete [] this->prefactors;
    delete [] this->buffers;
}

template<typename T>
void sphericart::SphericalHarmonics<T>::compute(const std::vector<T>& xyz, std::vector<T>& sph) {
    auto n_samples = xyz.size() / 3;
    sph.resize(n_samples * (l_max + 1) * (l_max + 1));

    if (xyz.size() == 3) {
        this->compute_sample(xyz.data(), sph.data());
    } else {
        this->compute_array(xyz.size() / 3, xyz.data(), sph.data());
    }
}

template<typename T>
void sphericart::SphericalHarmonics<T>::compute(const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph) {
    auto n_samples = xyz.size() / 3;
    sph.resize(n_samples * (l_max + 1) * (l_max + 1));
    dsph.resize(n_samples * 3 * (l_max + 1) * (l_max + 1));

    if (xyz.size() == 3) {
        this->compute_sample(xyz.data(), sph.data(), dsph.data());
    } else {
        this->compute_array(xyz.size() / 3, xyz.data(), sph.data(), dsph.data());
    }
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::SphericalHarmonics<float>;
template class sphericart::SphericalHarmonics<double>;


template<typename T>
std::pair<std::vector<T>, std::vector<T> > sphericart::spherical_harmonics(size_t l_max, const std::vector<T>& xyz, bool normalized) {
    auto calculator = sphericart::SphericalHarmonics<T>(l_max, normalized);
    auto sph = std::vector<T>();
    auto dsph = std::vector<T>();

    calculator.compute(xyz, sph, dsph);
    return std::make_pair(sph, dsph);
}

template std::pair<std::vector<double>, std::vector<double> > sphericart::spherical_harmonics(size_t l_max, const std::vector<double>& xyz, bool normalized);
template std::pair<std::vector<float>, std::vector<float> > sphericart::spherical_harmonics(size_t l_max, const std::vector<float>& xyz, bool normalized);
