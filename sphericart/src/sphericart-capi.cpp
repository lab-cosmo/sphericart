#include "sphericart.hpp"
#include "sphericart.h"

extern "C" sphericart_calculator_t *sphericart_new(size_t l_max, bool normalized) {
    return new sphericart::SphericalHarmonics<double>(l_max, normalized);
}

extern "C" void sphericart_delete(sphericart_calculator_t* calculator) {
    delete calculator;
}

extern "C" void sphericart_compute_array(sphericart_calculator_t* calculator, size_t n_samples, const double* xyz, double* sph, double* dsph) {
    if (dsph == nullptr) {
        calculator->compute_array(static_cast<int>(n_samples), xyz, sph);
    } else {
        calculator->compute_array(static_cast<int>(n_samples), xyz, sph, dsph);
    }
}

extern "C" void sphericart_compute_sample(sphericart_calculator_t* calculator, const double* xyz, double* sph, double* dsph) {
    if (dsph == nullptr) {
        calculator->compute_sample(xyz, sph);
    } else {
        calculator->compute_sample(xyz, sph, dsph);
    }
}

extern "C" sphericart_calculator_f_t* sphericart_new_f(size_t l_max, bool normalized) {
    return new sphericart::SphericalHarmonics<float>(l_max, normalized);
}

extern "C" void sphericart_compute_array_f(sphericart_calculator_f_t* calculator, size_t n_samples, const float* xyz, float* sph, float* dsph) {
    if (dsph == nullptr) {
        calculator->compute_array(static_cast<int>(n_samples), xyz, sph);
    } else {
        calculator->compute_array(static_cast<int>(n_samples), xyz, sph, dsph);
    }
}

extern "C" void sphericart_compute_sample_f(sphericart_calculator_f_t* calculator, const float* xyz, float* sph, float* dsph) {
    if (dsph == nullptr) {
        calculator->compute_sample(xyz, sph);
    } else {
        calculator->compute_sample(xyz, sph, dsph);
    }
}

extern "C" void sphericart_delete_f(sphericart_calculator_f_t* calculator) {
    delete calculator;
}
