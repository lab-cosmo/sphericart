#include <cmath>
#include<iostream>
#include "sphericart.hpp"
#include "templates.hpp"

template<typename DTYPE>
inline void _compute_sph_prefactors(int l_max, DTYPE *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
        Use an iterative formula to avoid computing a ratio
        of factorials, and incorporates the 1/sqrt(2) that
        is associated with the Yl0's
        Also computes a set of coefficients that are needed
        in the iterative calculation of the Qlm, and just
        stashes them at the end of factors, which should therefore
        be (l_max+1)*(l_max+2) in size
    */

    auto k = 0; // quick access index
    for (int l = 0; l <= l_max; ++l) {
        DTYPE factor = (2 * l + 1) / (2 * M_PI);
        // incorporates  the 1/sqrt(2) that goes with the m=0 SPH
        factors[k] = sqrt(factor) * M_SQRT1_2;
        for (int m = 1; m <= l; ++m) {
            factor *= 1.0 / (l * (l + 1) + m * (1 - m));
            if (m % 2 == 0) {
                factors[k + m] = sqrt(factor);
            } else {
                factors[k + m] = -sqrt(factor);
            }
        }
        k += l + 1;
    }

    // that are needed in the recursive calculation of Qlm.
    // Xll is just Qll, Xlm is the factor that enters the alternative m recursion 
    factors[k] = 1.0; k += 1;
    for (int l = 1; l < l_max + 1; l++) {
        factors[k+l] = -(2 * l - 1) * factors[k - 1];
        for (int m = l - 1; m >= 0; --m) {
            factors[k + m] = -1.0 / ((l + m + 1) * (l - m));
        }        
        k += l + 1;
    }
}

// specialized versions
void sphericart::compute_sph_prefactors(int l_max, double *factors) {
    _compute_sph_prefactors<double>(l_max, factors);
}

/*
void sphericart::compute_sph_prefactors(int l_max, float *factors) {
    _compute_sph_prefactors<float>(l_max, factors);
}
*/

template <typename DTYPE, bool DO_DERIVATIVES, bool NORMALIZED>
inline void _hardcoded_lmax_sample_switch(int n_samples, int l_max, const DTYPE *xyz, DTYPE *sph, DTYPE *dsph) {
    switch (l_max) {
    case 0:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 0>(n_samples, xyz, sph, dsph);
        break;
    case 1:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 1>(n_samples, xyz, sph, dsph);
        break;
    case 2:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 2>(n_samples, xyz, sph, dsph);
        break;
    case 3:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 3>(n_samples, xyz, sph, dsph);
        break;
    case 4:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 4>(n_samples, xyz, sph, dsph);
        break;
    case 5:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 5>(n_samples, xyz, sph, dsph);
        break;
    case 6:
        hardcoded_sph_sample<DTYPE, DO_DERIVATIVES, NORMALIZED, 6>(n_samples, xyz, sph, dsph);
        break;
    }
}

template <typename DTYPE, bool DO_DERIVATIVES, bool NORMALIZED>
inline void _hardcoded_lmax_switch(int n_samples, int l_max, const DTYPE *xyz, DTYPE *sph, DTYPE *dsph) {
    switch (l_max) {
    case 0:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 0>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 1:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 1>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 2:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 2>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 3:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 3>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 4:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 4>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 5:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 5>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    case 6:
        hardcoded_sph<DTYPE, DO_DERIVATIVES, NORMALIZED, 6>(n_samples, 0, nullptr, nullptr, xyz, sph, dsph);
        break;
    }
}

template<typename DTYPE>
inline void _cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const DTYPE *prefactors,
    const DTYPE *xyz,
    DTYPE *sph,
    DTYPE *dsph
) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the
        r^l term can be easily incorporated into any radial function or
        added a posteriori (with the corresponding derivative).
    */

    // call directly the fast ones
    if (l_max <= SPHERICART_LMAX_HARDCODED) {
        if (dsph == nullptr) {
            _hardcoded_lmax_switch<DTYPE, false,false>(n_samples, l_max, xyz, sph, dsph);
        } else {
            _hardcoded_lmax_switch<DTYPE, true, false>(n_samples, l_max, xyz, sph, dsph);
        }
    } else {
#ifdef _OPENMP
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3*omp_get_max_threads()];
#else
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3];
#endif                     
        if (dsph == nullptr) {
            generic_sph<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        } else {
            generic_sph<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        }
    }
}

template<typename DTYPE>
inline void _cartesian_spherical_harmonics_sample(
    int n_samples,
    int l_max,
    const DTYPE *prefactors,
    const DTYPE *xyz,
    DTYPE *sph,
    DTYPE *dsph
) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the
        r^l term can be easily incorporated into any radial function or
        added a posteriori (with the corresponding derivative).
    */

    // call directly the fast ones
    if (l_max <= SPHERICART_LMAX_HARDCODED) {
        if (dsph == nullptr) {
            _hardcoded_lmax_switch<DTYPE, false,false>(n_samples, l_max, xyz, sph, dsph);
        } else {
            _hardcoded_lmax_switch<DTYPE, true, false>(n_samples, l_max, xyz, sph, dsph);
        }
    } else {
#ifdef _OPENMP
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3*omp_get_max_threads()];
#else
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3];
#endif             
        if (dsph == nullptr) {
            generic_sph<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        } else {
            generic_sph<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        }
    }
}

void sphericart::cartesian_spherical_harmonics(int n_samples,
    int l_max, const double *prefactors,
    const double *xyz, double *sph, double *dsph) {
    _cartesian_spherical_harmonics<double>(n_samples, l_max, prefactors, xyz, sph, dsph);
}

/*
void sphericart::cartesian_spherical_harmonics(int n_samples,
    int l_max, const float *prefactors,
    const float *xyz, float *sph, float *dsph) {
    _cartesian_spherical_harmonics<float>(n_samples, l_max, prefactors, xyz, sph, dsph);
}
*/

template<typename DTYPE>
inline void _normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const DTYPE *prefactors,
    const DTYPE *xyz,
    DTYPE *sph,
    DTYPE *dsph
) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the
        r^l term can be easily incorporated into any radial function or
        added a posteriori (with the corresponding derivative).
    */
     
    // call directly the fast ones
    if (l_max <= SPHERICART_LMAX_HARDCODED) {
        if (dsph == nullptr) {
            _hardcoded_lmax_switch<DTYPE, false,true>(n_samples, l_max, xyz, sph, dsph);
        } else {
            _hardcoded_lmax_switch<DTYPE, true, true>(n_samples, l_max, xyz, sph, dsph);
        }
    } else {
#ifdef _OPENMP
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3*omp_get_max_threads()];
#else
        DTYPE *buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3];
#endif
        if (dsph == nullptr) {
            generic_sph<DTYPE, false, true, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        } else {
            generic_sph<DTYPE, true, true, SPHERICART_LMAX_HARDCODED>(n_samples, l_max, prefactors, buffers, xyz, sph, dsph);
        }
    }
}

void sphericart::normalized_spherical_harmonics(
    int n_samples, int l_max,
    const double *prefactors, 
    const double *xyz, double *sph, double *dsph) {
    _normalized_spherical_harmonics<double>(n_samples, l_max, prefactors, xyz, sph, dsph);
}

/*
void sphericart::normalized_spherical_harmonics(
    int n_samples, int l_max,
    const float *prefactors, 
    const float *xyz, float *sph, float *dsph) {
    _normalized_spherical_harmonics<float>(n_samples, l_max, prefactors, xyz, sph, dsph);
}
*/

// macro to define different possible hardcoded function calls
#define _HARCODED_SWITCH_CASE(L_MAX) \
    if (this->normalize) { \
        this->_array_no_derivatives = &hardcoded_sph<DTYPE, false, true, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<DTYPE, true, true, L_MAX>; \
    } else { \
        this->_array_no_derivatives = &hardcoded_sph<DTYPE, false, false, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<DTYPE, true, false, L_MAX>; \
    }

template<typename DTYPE>
sphericart::SphericalHarmonics<DTYPE>::SphericalHarmonics(size_t l_max, bool normalize) {
    this->l_max = l_max;
    this->normalize = normalize;
    this->prefactors = new DTYPE[(l_max+1)*(l_max+2)];
    
    // buffers for cos, sin, 2mz arrays
#ifdef _OPENMP
    // allocates buffers that are large enough to store thread-local data
    this->buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3*omp_get_max_threads()];
#else
    this->buffers = new DTYPE[(l_max+1)*(l_max+2)/2*3];
#endif
    _compute_sph_prefactors<DTYPE>((int) l_max, this->prefactors);

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
        if (this->normalize) {
            this->_array_no_derivatives = &generic_sph<DTYPE, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<DTYPE, true, true, SPHERICART_LMAX_HARDCODED>;
        } else {
            this->_array_no_derivatives = &generic_sph<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>;
        }
    }
}

template<typename DTYPE>
sphericart::SphericalHarmonics<DTYPE>::~SphericalHarmonics() {
    delete [] this->prefactors;
    delete [] this->buffers;
}

template<typename DTYPE>
void sphericart::SphericalHarmonics<DTYPE>::compute(size_t n_samples, const std::vector<DTYPE>& xyz, std::vector<DTYPE>& sph) {
    this->_array_no_derivatives(n_samples, this->l_max, this->prefactors, this->buffers, xyz.data(), sph.data(), nullptr);
}


template<typename DTYPE>
void sphericart::SphericalHarmonics<DTYPE>::compute(size_t n_samples, const std::vector<DTYPE>& xyz, std::vector<DTYPE>& sph, std::vector<DTYPE>& dsph) {
    this->_array_with_derivatives(n_samples, this->l_max, this->prefactors, this->buffers, xyz.data(), sph.data(), dsph.data());        
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::SphericalHarmonics<float>;
template class sphericart::SphericalHarmonics<double>;
