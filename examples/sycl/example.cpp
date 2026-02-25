/** @file example.cu
 *  @brief Usage example for the CUDA C++ API
 */

#include "sphericart_sycl.hpp"
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef DTYPE
#define DTYPE double
#endif

#include "sphericart.hpp"
#define TOLERANCE 1e-4 // High tolerance: finite differences are inaccurate for second
int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 1000;
    size_t l_max = 6;

    int size2;

    // initializes samples
    auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (DTYPE)rand() / (DTYPE)RAND_MAX * 2.0 - 1.0;
#ifdef PRINT_DEBUG
        std::cout << "xyz[" << i << "]: " << xyz[i] << std::endl;
#endif
    }

    // the class is templated, so one can also use 32-bit DTYPE operations
    auto xyz_f = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_f[i] = (DTYPE)xyz[i];
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, one also can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    // auto ddsph = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);
    std::fill(sph.begin(), sph.end(), 0.0); //
    std::fill(dsph.begin(), dsph.end(), 0.0);

    // initializes local vectors
    auto sph_f = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    // auto ddsph_f = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);
    std::fill(sph_f.begin(), sph_f.end(), 0.0);
    std::fill(dsph_f.begin(), dsph_f.end(), 0.0);
    // std::fill(ddsph_f.begin(), ddsph_f.end(), 0.0);

    if (1) { // scope 1
             //  initializes device vectors
        DEVICE_INIT(DTYPE, xyz_device, xyz_f.data(), xyz_f.size())
        DEVICE_INIT(DTYPE, sph_device, sph_f.data(), sph_f.size())
        DEVICE_INIT(DTYPE, dsph_device, dsph_f.data(), dsph_f.size())
        // DEVICE_INIT(DTYPE, ddsph_device, ddsph_f.data(), ddsph_f.size())
        /* ===== API calls ===== */

        // no gradients

        // internal buffers and numerical factors are initalized at construction
        // SYCL calculator
        sphericart::sycl::SphericalHarmonics<DTYPE> calculator_sycl(l_max);
        // CPU calculator
        auto calculator = sphericart::SphericalHarmonics<DTYPE>(l_max);
        //
        //    // allcate device memory
        //    // calculation examples
        // calculator_sycl.compute(xyz_device, n_samples, sph_device); // no gradients
        // calculator.compute(xyz, sph);
        std::cout << "ROUND 1 \n";
        std::cout << "    Computing SPH only \n";
        calculator.compute(xyz, sph);                               // with gradients
        calculator_sycl.compute(xyz_device, n_samples, sph_device); // with gradients00

        // Get results from device to host
        DEVICE_GET(DTYPE, sph_f.data(), sph_device, sph_f.size())

        /* ===== check results ===== */
        size2 =
            (l_max + 1) * (l_max + 1); // Size of the second+third dimensions in derivative arrays
        DTYPE sph_error = 0.0, sph_norm = 0.0, sph_error_h = 0.0, sph_error_h_max = 0.0;
        for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
            sph_error_h = (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
            sph_error += sph_error_h;
            if (sph_error_h > sph_error_h_max) {
                sph_error_h_max = sph_error_h;
            }
            sph_norm += sph[i] * sph[i];
#ifdef PRINT_DEBUG
            std::cout << "sph_f[" << i << "]: " << sph_f[i] << ", sph[" << i << "]: " << sph[i]
                      << std::endl;
#endif
            // printf("SPHERR: %e ,SPHERNOR %e\n", sph_f[i] - sph[i], sph_norm);
        }

#if 1
        std::cout << "CPU vs GPU relative error SPH: " << std::scientific << std::setprecision(8)
                  << sqrt(sph_error / sph_norm) << std::endl;
        std::cout << "Maximum squared error SPH: " << std::scientific << std::setprecision(8)
                  << sph_error_h_max << std::endl;
        std::cout << std::endl;
#endif
    }

    std::fill(sph_f.begin(), sph_f.end(), 0.0);
    std::fill(dsph_f.begin(), dsph_f.end(), 0.0);
    std::fill(sph.begin(), sph.end(), 0.0);
    std::fill(dsph.begin(), dsph.end(), 0.0);

    { // scope 2

        DEVICE_INIT(DTYPE, xyz_device, xyz_f.data(), xyz_f.size())
        DEVICE_INIT(DTYPE, sph_device, sph_f.data(), sph_f.size())
        DEVICE_INIT(DTYPE, dsph_device, dsph_f.data(), dsph_f.size())

        std::cout << "ROUND 2 \n";
        std::cout << "    Computing gradients (SPH and DSPH) \n";

        // internal buffers and numerical factors are initalized at construction
        sphericart::sycl::SphericalHarmonics<DTYPE> calculator_sycl(l_max);
        auto calculator = sphericart::SphericalHarmonics<DTYPE>(l_max);

        calculator.compute_with_gradients(xyz, sph, dsph); // compute with gradients in host
        calculator_sycl.compute_with_gradients(
            xyz_device, n_samples, sph_device, dsph_device
        ); // compute with gradients in device
        // Get results from device to host
        DEVICE_GET(DTYPE, sph_f.data(), sph_device, sph_f.size())
        DEVICE_GET(DTYPE, dsph_f.data(), dsph_device, dsph_f.size());

        /* ===== check results ===== */

        DTYPE size2 =
            (l_max + 1) * (l_max + 1); // Size of the second+third dimensions in derivative arrays
        DTYPE sph_error = 0.0, sph_norm = 0.0, sph_error_h = 0.0, sph_error_h_max = 0.0;
        // sph_error = 0.0, sph_norm = 0.0, sph_error_h = 0.0, sph_error_h_max = 0.0;
        for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
            sph_error_h = (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
            sph_error += sph_error_h;
            if (sph_error_h > sph_error_h_max) {
                sph_error_h_max = sph_error_h;
            }
            sph_norm += sph[i] * sph[i];
#ifdef PRINT_DEBUG
            std::cout << "sph_f[" << i << "]: " << sph_f[i] << ", sph[" << i << "]: " << sph[i]
                      << std::endl;
#endif
            // printf("SPHERR: %e ,SPHERNOR %e\n", sph_f[i] - sph[i], sph_norm);
        }

#if 1
        std::cout << "CPU vs GPU relative error SPH: " << std::scientific << std::setprecision(8)
                  << sqrt(sph_error / sph_norm) << std::endl;
        std::cout << "Maximum squared error SPH: " << std::scientific << std::setprecision(8)
                  << sph_error_h_max << std::endl;
        std::cout << std::endl;
#endif

        /* ===== check results ===== */

        DTYPE dsph_error = 0.0, dsph_norm = 0.0, dsph_error_h = 0.0, dsph_error_h_max = 0.0;
        int n_sph = (l_max + 1) * (l_max + 1);
        for (size_t alpha = 0; alpha < 3; alpha++) {
            for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
                for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                    DTYPE d0 = dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                    DTYPE d0_f = dsph_f[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                    dsph_error_h = (d0 - d0_f) * (d0 - d0_f);
                    if (dsph_error_h > dsph_error_h_max) {
                        dsph_error_h_max = dsph_error_h;
                    }
                    dsph_error += dsph_error_h;
                    dsph_norm += d0 * d0;
#ifdef PRINT_DEBUG
                    std::cout << "d0: " << d0 << ", d0_f: " << d0_f << std::endl;
                    if (std::abs(d0 / d0_f - 1.0) > TOLERANCE) {
                        std::cout << "Wrong first derivative: " << d0 << " vs " << d0_f << std::endl;
                    }
#endif
                }
            }
        }
#if 1
        std::cout << "CPU vs GPU relative error DSPH: " << std::scientific << std::setprecision(8)
                  << sqrt(dsph_error / dsph_norm) << std::endl;
        std::cout << "Maximum squared error DSPH: " << std::scientific << std::setprecision(8)
                  << dsph_error_h_max << std::endl;
        std::cout << std::endl;
#endif
        //
    }
    return 0;
}
