#include <cmath>
#include <iostream>
#include "sphericart.hpp"
#include "templates.hpp"

template<typename DTYPE>
void sphericart::compute_sph_prefactors(int l_max, DTYPE *factors) {
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

template void sphericart::compute_sph_prefactors<double>(int l_max, double *factors); 
template void sphericart::compute_sph_prefactors<float>(int l_max, float *factors); 

// macro to define different possible hardcoded function calls
#define _HARCODED_SWITCH_CASE(L_MAX) \
    if (this->normalized) { \
        this->_array_no_derivatives = &hardcoded_sph<DTYPE, false, true, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<DTYPE, true, true, L_MAX>; \
        this->_sample_no_derivatives = &hardcoded_sph_sample<DTYPE, false, true, SPHERICART_LMAX_HARDCODED>; \
        this->_sample_with_derivatives = &hardcoded_sph_sample<DTYPE, true, true, SPHERICART_LMAX_HARDCODED>; \
    } else { \
        this->_array_no_derivatives = &hardcoded_sph<DTYPE, false, false, L_MAX>; \
        this->_array_with_derivatives = &hardcoded_sph<DTYPE, true, false, L_MAX>; \
        this->_sample_no_derivatives = &hardcoded_sph_sample<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>; \
        this->_sample_with_derivatives = &hardcoded_sph_sample<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>; \
    }

template<typename DTYPE>
sphericart::SphericalHarmonics<DTYPE>::SphericalHarmonics(size_t l_max, bool normalized) {
    this->l_max = (int) l_max;
    this->size_y = (int) (l_max + 1) * (l_max + 1);
    this->size_q = (int) (l_max + 1) * (l_max + 2) / 2;
    this->normalized = normalized;
    this->prefactors = new DTYPE[(l_max+1)*(l_max+2)];
    
    // buffers for cos, sin, 2mz arrays
#ifdef _OPENMP
    // allocates buffers that are large enough to store thread-local data
    this->buffers = new DTYPE[this->size_q*3*omp_get_max_threads()];
#else
    this->buffers = new DTYPE[this->size_q*3];
#endif
    compute_sph_prefactors<DTYPE>((int) l_max, this->prefactors);

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
            this->_array_no_derivatives = &generic_sph<DTYPE, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<DTYPE, true, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives = &generic_sph_sample<DTYPE, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives = &generic_sph_sample<DTYPE, true, true, SPHERICART_LMAX_HARDCODED>;
        } else {
            this->_array_no_derivatives = &generic_sph<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives = &generic_sph<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives = &generic_sph_sample<DTYPE, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives = &generic_sph_sample<DTYPE, true, false, SPHERICART_LMAX_HARDCODED>;
        }
    }
}

template<typename DTYPE>
sphericart::SphericalHarmonics<DTYPE>::~SphericalHarmonics() {
    delete [] this->prefactors;
    delete [] this->buffers;
}

template<typename DTYPE>
void sphericart::SphericalHarmonics<DTYPE>::compute(const std::vector<DTYPE>& xyz, std::vector<DTYPE>& sph) {
    if (xyz.size()==3) {
        this->compute_sample(xyz.data(), sph.data());
    } else {
        this->compute_array(xyz.size()/3, xyz.data(), sph.data());
    }
}

template<typename DTYPE>
void sphericart::SphericalHarmonics<DTYPE>::compute(const std::vector<DTYPE>& xyz, std::vector<DTYPE>& sph, std::vector<DTYPE>& dsph) {
    if (xyz.size()==3) {
        this->compute_sample(xyz.data(), sph.data(), dsph.data());        
    } else {
        this->compute_array(xyz.size()/3, xyz.data(), sph.data(), dsph.data());
    }     
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::SphericalHarmonics<float>;
template class sphericart::SphericalHarmonics<double>;


template<typename DTYPE>
std::pair<std::vector<DTYPE>, std::vector<DTYPE> > sphericart::spherical_harmonics(size_t l_max, const std::vector<DTYPE>& xyz, bool normalized) {
    
    auto sph_class = sphericart::SphericalHarmonics<DTYPE>(l_max, normalized);
    std::vector<DTYPE> sph((xyz.size()/3)*(l_max+1)*(l_max+1));
    std::vector<DTYPE> dsph(xyz.size()*(l_max+1)*(l_max+1));
    sph_class.compute(xyz, sph, dsph);
    return std::make_pair(sph, dsph);
}

template std::pair<std::vector<double>, std::vector<double> > sphericart::spherical_harmonics(size_t l_max, const std::vector<double>& xyz, bool normalized);
template std::pair<std::vector<float>, std::vector<float> > sphericart::spherical_harmonics(size_t l_max, const std::vector<float>& xyz, bool normalized);
