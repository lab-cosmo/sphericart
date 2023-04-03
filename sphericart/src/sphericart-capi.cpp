#include <stdexcept>

#include "sphericart.hpp"
#include "sphericart.h"

extern "C" sphericart_calculator_t *sphericart_new(size_t l_max, bool normalized) {
    try {
        return new sphericart::SphericalHarmonics<double>(l_max, normalized);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_delete(sphericart_calculator_t* calculator) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_compute_array(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        if (dsph == nullptr) {
            calculator->compute_array(xyz, xyz_length, sph, sph_length);
        } else {
            calculator->compute_array(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
        }
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_compute_sample(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        if (dsph == nullptr) {
            calculator->compute_sample(xyz, xyz_length, sph, sph_length);
        } else {
            calculator->compute_sample(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
        }
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" sphericart_calculator_f_t* sphericart_new_f(size_t l_max, bool normalized) {
     try {
        return new sphericart::SphericalHarmonics<float>(l_max, normalized);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_delete_f(sphericart_calculator_f_t* calculator) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_compute_array_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        if (dsph == nullptr) {
            calculator->compute_array(xyz, xyz_length, sph, sph_length);
        } else {
            calculator->compute_array(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
        }
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_compute_sample_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        if (dsph == nullptr) {
            calculator->compute_sample(xyz, xyz_length, sph, sph_length);
        } else {
            calculator->compute_sample(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
        }
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}
