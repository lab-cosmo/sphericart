/** @file example.cpp
 *  @brief Usage example for the C++ API
 */

#include <cmath>
#include <cstdio>

#include "sphericart.hpp"

int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 10000;
    size_t l_max = 10;

    // initializes samples
    auto xyz = std::vector<double>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, however one can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<double>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<double>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph = std::vector<double>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // the class is templated, so one can also use 32-bit float operations
    auto xyz_f = std::vector<float>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_f[i] = (float)xyz[i];
    }
    auto sph_f = std::vector<float>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<float>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_f = std::vector<float>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // the class can be used to compute the for a full arrays of points (as
    // above) or on individual samples - this is deduced from the size of the
    // array
    auto xyz_sample = std::vector<double>(3, 0.0);
    auto sph_sample = std::vector<double>((l_max + 1) * (l_max + 1), 0.0);
    auto dsph_sample = std::vector<double>(3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_sample = std::vector<double>(3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    auto calculator = sphericart::SphericalHarmonics<double>(l_max);

    calculator.compute(xyz, sph);                      // no gradients
    calculator.compute_with_gradients(xyz, sph, dsph); // gradients
    calculator.compute_with_hessians(xyz, sph, dsph,
                                     ddsph); // gradients and hessians

    // the single-sample evaluation provides direct access to the main
    // calculator, avoiding the loop over samples and allowing e.g. custom
    // parallelization
    calculator.compute(xyz_sample, sph_sample); // no gradients
    calculator.compute_with_gradients(xyz_sample, sph_sample,
                                      dsph_sample); // gradients
    calculator.compute_with_hessians(
        xyz_sample,
        sph_sample,
        dsph_sample,
        ddsph_sample
    ); // gradients and hessians

    // float version
    auto calculator_f = sphericart::SphericalHarmonics<float>(l_max);
    calculator_f.compute(xyz_f, sph_f);                        // no gradients
    calculator_f.compute_with_gradients(xyz_f, sph_f, dsph_f); // gradients
    calculator_f.compute_with_hessians(xyz_f, sph_f, dsph_f,
                                       ddsph_f); // gradients and hessians

    /* ===== check results ===== */

    double sph_error = 0.0, sph_norm = 0.0;
    for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
        sph_error += (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
        sph_norm += sph[i] * sph[i];
    }
    printf("Float vs double relative error: %12.8e\n", sqrt(sph_error / sph_norm));

    return 0;
}
