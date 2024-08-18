/** @file test_hardcoding.cpp
 *  @brief Checks consistency of array and sample calls
 */

#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iostream>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

#define _SPH_TOL 1e-9
#define DTYPE double
using namespace sphericart;

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    size_t MAX_L_VALUE = 10;

    size_t n_samples = 2;
    auto xyz_sample = std::vector<DTYPE>({1., 2., 3.});
    auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; i += 3) {
        xyz[i] = xyz_sample[0];
        xyz[i + 1] = xyz_sample[1];
        xyz[i + 2] = xyz_sample[2];
    }

    bool test_passed = true;
    for (size_t l_max = 0; l_max <= MAX_L_VALUE; l_max++) {
        auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
        auto dsph = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
        auto sph_sample = std::vector<DTYPE>(1 * (l_max + 1) * (l_max + 1), 0.0);
        auto dsph_sample = std::vector<DTYPE>(1 * 3 * (l_max + 1) * (l_max + 1), 0.0);
        SphericalHarmonics<DTYPE> SH(l_max);
        SH.compute_with_gradients(xyz_sample, sph_sample, dsph_sample);
        SH.compute_with_gradients(xyz, sph, dsph);
        int size3 = 3 * (l_max + 1) * (l_max + 1); // Size of the third dimension in derivative
                                                   // arrays (or second in normal sph arrays).
        int size2 = (l_max + 1) * (l_max + 1);     // Size of the second+third
                                                   // dimensions in derivative arrays
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            for (size_t l = 0; l < (l_max + 1); l++) {
                for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
                    if (fabs(sph[size2 * i_sample + l * l + l + m] / sph_sample[l * l + l + m] - 1) >
                        _SPH_TOL) {
                        printf(
                            "Mismatch detected at i_sample = %zu, L = %zu, "
                            "m = %d \n",
                            i_sample,
                            l,
                            m
                        );
                        printf(
                            "SPH: %e, %e\n",
                            sph[size2 * i_sample + l * l + l + m],
                            sph_sample[l * l + l + m]
                        );
                        test_passed = false;
                    }
                    if (fabs(
                            dsph[size3 * i_sample + size2 * 0 + l * l + l + m] /
                                dsph_sample[size2 * 0 + l * l + l + m] -
                            1
                        ) > _SPH_TOL) {
                        printf(
                            "Mismatch detected at i_sample = %zu, L = %zu, "
                            "m = %d \n",
                            i_sample,
                            l,
                            m
                        );
                        printf(
                            "DxSPH: %e, %e\n",
                            dsph[size3 * i_sample + size2 * 0 + l * l + l + m],
                            dsph_sample[size2 * 0 + l * l + l + m]
                        );
                        test_passed = false;
                    }
                    if (fabs(
                            dsph[size3 * i_sample + size2 * 1 + l * l + l + m] /
                                dsph_sample[size2 * 1 + l * l + l + m] -
                            1
                        ) > _SPH_TOL) {
                        printf(
                            "Mismatch detected at i_sample = %zu, L = %zu, "
                            "m = %d \n",
                            i_sample,
                            l,
                            m
                        );
                        printf(
                            "DySPH: %e, %e\n",
                            dsph[size3 * i_sample + size2 * 1 + l * l + l + m],
                            dsph_sample[size2 * 1 + l * l + l + m]
                        );
                        test_passed = false;
                    }
                    if (fabs(
                            dsph[size3 * i_sample + size2 * 2 + l * l + l + m] /
                                dsph_sample[size2 * 2 + l * l + l + m] -
                            1
                        ) > _SPH_TOL) {
                        printf(
                            "Mismatch detected at i_sample = %zu, L = %zu, "
                            "m = %d \n",
                            i_sample,
                            l,
                            m
                        );
                        printf(
                            "DzSPH: %e, %e\n",
                            dsph[size3 * i_sample + size2 * 1 + l * l + l + m],
                            dsph_sample[size2 * 2 + l * l + l + m]
                        );
                        test_passed = false;
                    }
                }
            }
        }
    }
    if (test_passed) {
        printf("Consistency test passed\n");
        return 0;
    } else {
        printf("Consistency test failed\n");
        return -1;
    }
}
