#include <stdexcept>

#include "sphericart.h"
#include "sphericart.hpp"

extern "C" sphericart_spherical_harmonics_calculator_t* sphericart_spherical_harmonics_new(
    size_t l_max
) {
    try {
        return new sphericart::SphericalHarmonics<double>(l_max);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_spherical_harmonics_delete(
    sphericart_spherical_harmonics_calculator_t* calculator
) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
) {
    try {
        calculator->compute_array(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array_with_gradients(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array_with_hessians(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_array_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
) {
    try {
        calculator->compute_sample(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample_with_gradients(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_sample_with_gradients(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample_with_hessians(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_sample_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" sphericart_spherical_harmonics_calculator_f_t* sphericart_spherical_harmonics_new_f(
    size_t l_max
) {
    try {
        return new sphericart::SphericalHarmonics<float>(l_max);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_spherical_harmonics_delete_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator
) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
) {
    try {
        calculator->compute_array(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array_with_gradients_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_array_with_hessians_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_array_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
) {
    try {
        calculator->compute_sample(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample_with_gradients_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_sample_with_gradients(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_spherical_harmonics_compute_sample_with_hessians_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_sample_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" int sphericart_spherical_harmonics_omp_num_threads(
    sphericart_spherical_harmonics_calculator_t* calculator
) {
    return calculator->get_omp_num_threads();
}

extern "C" int sphericart_spherical_harmonics_omp_num_threads_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator
) {
    return calculator->get_omp_num_threads();
}

extern "C" sphericart_solid_harmonics_calculator_t* sphericart_solid_harmonics_new(size_t l_max) {
    try {
        return new sphericart::SolidHarmonics<double>(l_max);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_solid_harmonics_delete(sphericart_solid_harmonics_calculator_t* calculator) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_solid_harmonics_compute_array(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
) {
    try {
        calculator->compute_array(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_array_with_gradients(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_array_with_hessians(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_array_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
) {
    try {
        calculator->compute_sample(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample_with_gradients(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_sample_with_gradients(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample_with_hessians(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_sample_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" sphericart_solid_harmonics_calculator_f_t* sphericart_solid_harmonics_new_f(size_t l_max) {
    try {
        return new sphericart::SolidHarmonics<float>(l_max);
    } catch (...) {
        // TODO: better error handling
        return nullptr;
    }
}

extern "C" void sphericart_solid_harmonics_delete_f(
    sphericart_solid_harmonics_calculator_f_t* calculator
) {
    try {
        delete calculator;
    } catch (...) {
        // nothing to do
    }
}

extern "C" void sphericart_solid_harmonics_compute_array_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
) {
    try {
        calculator->compute_array(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_array_with_gradients_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_length, dsph, dsph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_array_with_hessians_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_array_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
) {
    try {
        calculator->compute_sample(xyz, xyz_length, sph, sph_length);
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample_with_gradients_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
) {
    try {
        calculator->compute_sample_with_gradients(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" void sphericart_solid_harmonics_compute_sample_with_hessians_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
) {
    try {
        calculator->compute_sample_with_hessians(
            xyz, xyz_length, sph, sph_length, dsph, dsph_length, ddsph, ddsph_length
        );
    } catch (const std::exception& e) {
        // TODO: better error handling
        printf("fatal error: %s\n", e.what());
        abort();
    } catch (...) {
        printf("fatal error: unknown exception type\n");
        abort();
    }
}

extern "C" int sphericart_solid_harmonics_omp_num_threads(
    sphericart_solid_harmonics_calculator_t* calculator
) {
    return calculator->get_omp_num_threads();
}

extern "C" int sphericart_solid_harmonics_omp_num_threads_f(
    sphericart_solid_harmonics_calculator_f_t* calculator
) {
    return calculator->get_omp_num_threads();
}
