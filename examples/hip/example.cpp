/** @file example.cpp
 *  @brief Usage example for the HIP C++ API
 */

#include "sphericart_hip.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef DTYPE
#define DTYPE double
#endif

#include "sphericart.hpp"

#define CHECK_HIP(call)                                                       \
    do {                                                                      \
        hipError_t status_ = (call);                                          \
        if (status_ != hipSuccess) {                                          \
            std::cout << "HIP error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << hipGetErrorString(status_) << std::endl;     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

 #define TOLERANCE 1e-4 // High tolerance: finite differences are inaccurate for second

int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 1000;
    size_t l_max = 6;

    // initializes samples
    auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (DTYPE)rand() / (DTYPE)RAND_MAX * 2.0 - 1.0;
    }

    // host-side results, used as a reference
    auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // device-side results
    auto sph_f = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);

    DTYPE* xyz_device = nullptr;
    DTYPE* sph_device = nullptr;
    DTYPE* dsph_device = nullptr;
    CHECK_HIP(hipMalloc(&xyz_device, xyz.size() * sizeof(DTYPE)));
    CHECK_HIP(hipMalloc(&sph_device, sph.size() * sizeof(DTYPE)));
    CHECK_HIP(hipMalloc(&dsph_device, dsph.size() * sizeof(DTYPE)));
    CHECK_HIP(hipMemcpy(xyz_device, xyz.data(), xyz.size() * sizeof(DTYPE), hipMemcpyHostToDevice));

    /* ===== API calls ===== */

    // internal buffers and numerical factors are initialized at construction
    // HIP calculator
    sphericart::hip::SphericalHarmonics<DTYPE> calculator_hip(l_max);
    // CPU calculator
    auto calculator = sphericart::SphericalHarmonics<DTYPE>(l_max);

    std::cout << "ROUND 1 \n";
    std::cout << "    Computing SPH only \n";
    calculator.compute(xyz, sph);
    calculator_hip.compute(xyz_device, n_samples, sph_device);

    // Get results from device to host
    CHECK_HIP(hipMemcpy(sph_f.data(), sph_device, sph_f.size() * sizeof(DTYPE), hipMemcpyDeviceToHost));

    /* ===== check results ===== */
    DTYPE sph_error = 0.0, sph_norm = 0.0, sph_error_max = 0.0;
    for (size_t i = 0; i < sph.size(); ++i) {
        DTYPE error = (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
        sph_error += error;
        if (error > sph_error_max) {
            sph_error_max = error;
        }
        sph_norm += sph[i] * sph[i];
    }

    std::cout << "CPU vs GPU relative error SPH: " << std::scientific << std::setprecision(8)
              << sqrt(sph_error / sph_norm) << std::endl;
    std::cout << "Maximum squared error SPH: " << std::scientific << std::setprecision(8)
              << sph_error_max << std::endl;
    std::cout << std::endl;

    std::cout << "ROUND 2 \n";
    std::cout << "    Computing gradients (SPH and DSPH) \n";
    calculator.compute_with_gradients(xyz, sph, dsph);
    calculator_hip.compute_with_gradients(xyz_device, n_samples, sph_device, dsph_device);

    CHECK_HIP(hipMemcpy(sph_f.data(), sph_device, sph_f.size() * sizeof(DTYPE), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(dsph_f.data(), dsph_device, dsph_f.size() * sizeof(DTYPE), hipMemcpyDeviceToHost));

    /* ===== check results ===== */
    sph_error = 0.0, sph_norm = 0.0, sph_error_max = 0.0;
    for (size_t i = 0; i < sph.size(); ++i) {
        DTYPE error = (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
        sph_error += error;
        if (error > sph_error_max) {
            sph_error_max = error;
        }
        sph_norm += sph[i] * sph[i];
    }

    std::cout << "CPU vs GPU relative error SPH: " << std::scientific << std::setprecision(8)
              << sqrt(sph_error / sph_norm) << std::endl;
    std::cout << "Maximum squared error SPH: " << std::scientific << std::setprecision(8)
              << sph_error_max << std::endl;
    std::cout << std::endl;

    DTYPE dsph_error = 0.0, dsph_norm = 0.0, dsph_error_max = 0.0;
    int n_sph = (l_max + 1) * (l_max + 1);
    for (size_t alpha = 0; alpha < 3; alpha++) {
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                DTYPE d0 = dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                DTYPE d0_f = dsph_f[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                DTYPE error = (d0 - d0_f) * (d0 - d0_f);
                dsph_error += error;
                if (error > dsph_error_max) {
                    dsph_error_max = error;
                }
                dsph_norm += d0 * d0;
            }
        }
    }
    std::cout << "CPU vs GPU relative error DSPH: " << std::scientific << std::setprecision(8)
              << sqrt(dsph_error / dsph_norm) << std::endl;
    std::cout << "Maximum squared error DSPH: " << std::scientific << std::setprecision(8)
              << dsph_error_max << std::endl;
    std::cout << std::endl;

    CHECK_HIP(hipFree(xyz_device));
    CHECK_HIP(hipFree(sph_device));
    CHECK_HIP(hipFree(dsph_device));

    if (sqrt(sph_error / sph_norm) > TOLERANCE || sqrt(dsph_error / dsph_norm) > TOLERANCE) {
        std::cout << "Example failed: GPU results do not match the CPU reference" << std::endl;
        return 1;
    }

    return 0;
}
