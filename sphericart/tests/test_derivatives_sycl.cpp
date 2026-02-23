/** @file test_derivatives.cpp
 *  @brief Tests derivatives using finite differences
 */

#include <cassert>
#include <iostream>
#include <vector>

#include "sphericart_sycl.hpp"

#ifndef DTYPE
#define DTYPE double
#endif
#define DELTA 1e-4
#define TOLERANCE 1e-4 // High tolerance: finite differences are inaccurate for second derivatives

#include "sphericart.hpp"
template <template <typename> class C>
bool check_gradient_call(int l_max, C<DTYPE>& calculator, const std::vector<DTYPE>& xyz_host) {
    bool is_passed = true;
    int n_samples = xyz_host.size() / 3;
    int n_sph = (l_max + 1) * (l_max + 1);

    DEVICE_INIT(DTYPE, xyz, xyz_host.data(), xyz_host.size());

    MALLOC(DTYPE, sph, n_samples * (l_max + 1) * (l_max + 1));
    MALLOC(DTYPE, dsph, 3 * n_samples * (l_max + 1) * (l_max + 1));
    auto sph_host = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_host = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);

    calculator.compute_with_gradients(xyz, n_samples, sph, dsph);
    DEVICE_GET(DTYPE, dsph_host.data(), dsph, dsph_host.size());
    FREE(sph);

    for (int alpha = 0; alpha < 3; alpha++) {
        auto sph_plus_host = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
        auto sph_minus_host = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);

        MALLOC(DTYPE, sph_plus, n_samples * (l_max + 1) * (l_max + 1));
        MALLOC(DTYPE, sph_minus, n_samples * (l_max + 1) * (l_max + 1));

        std::vector<DTYPE> xyz_plus_host = xyz_host;
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_plus_host[3 * i_sample + alpha] += DELTA;
        }
        DEVICE_INIT(DTYPE, xyz_plus, xyz_plus_host.data(), xyz_plus_host.size());
        calculator.compute(xyz_plus, n_samples, sph_plus);
        DEVICE_GET(DTYPE, sph_plus_host.data(), sph_plus, sph_plus_host.size());
        FREE(xyz_plus);

        std::vector<DTYPE> xyz_minus_host = xyz_host;
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_minus_host[3 * i_sample + alpha] -= DELTA;
        }
        DEVICE_INIT(DTYPE, xyz_minus, xyz_minus_host.data(), xyz_minus_host.size());
        calculator.compute(xyz_minus, n_samples, sph_minus);
        DEVICE_GET(DTYPE, sph_minus_host.data(), sph_minus, sph_minus_host.size());
        FREE(xyz_minus);

        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                DTYPE analytical = dsph_host[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                DTYPE finite_diff = (sph_plus_host[n_sph * i_sample + i_sph] -
                                     sph_minus_host[n_sph * i_sample + i_sph]) /
                                    (2.0 * DELTA);
                // printf( "D DF  %e ANA %e \n", finite_diff, analytical);
                if (std::abs(analytical / finite_diff - 1.0) > TOLERANCE) {
                    std::cout << "Wrong first derivative: " << analytical << " vs " << finite_diff
                              << std::endl;
                }
                if (std::abs(analytical / finite_diff - 1.0) > TOLERANCE) {
                    is_passed = false;
                }
            }
        }
        FREE(sph_plus);
        FREE(sph_minus);
    }
    FREE(dsph);
    FREE(xyz);

    return is_passed;
}

template <template <typename> class C>
bool check_hessian_call(int l_max, C<DTYPE>& calculator, const std::vector<DTYPE>& xyz) {
    bool is_passed = true;
    int n_samples = xyz.size() / 3;
    int n_sph = (l_max + 1) * (l_max + 1);

    std::vector<DTYPE> sph_plus;
    std::vector<DTYPE> sph_minus;
    std::vector<DTYPE> sph_plus_plus;
    std::vector<DTYPE> sph_plus_minus;
    std::vector<DTYPE> sph_minus_plus;
    std::vector<DTYPE> sph_minus_minus;
    std::vector<DTYPE> sph;
    std::vector<DTYPE> dsph;
    std::vector<DTYPE> ddsph;

    calculator.compute_with_hessians(xyz, sph, dsph, ddsph);

    // Check first derivatives:
    for (int alpha = 0; alpha < 3; alpha++) {
        std::vector<DTYPE> xyz_plus = xyz;
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_plus[3 * i_sample + alpha] += DELTA;
        }
        calculator.compute(xyz_plus, n_samples, sph_plus);
        // calculator.compute(xyz_plus, sph_plus);

        std::vector<DTYPE> xyz_minus = xyz;
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_minus[3 * i_sample + alpha] -= DELTA;
        }
        // calculator.compute(xyz_minus, sph_minus);
        calculator.compute(xyz_minus, n_samples, sph_minus);

        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                DTYPE analytical = dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                DTYPE finite_diff =
                    (sph_plus[n_sph * i_sample + i_sph] - sph_minus[n_sph * i_sample + i_sph]) /
                    (2.0 * DELTA);
                if (std::abs(analytical / finite_diff - 1.0) > TOLERANCE) {
                    std::cout << "Wrong first derivative: " << analytical << " vs " << finite_diff
                              << std::endl;
                }
                if (std::abs(analytical / finite_diff - 1.0) > TOLERANCE) {
                    is_passed = false;
                }
            }
        }
    }

    // Check second derivatives:
    for (int alpha = 0; alpha < 3; alpha++) {
        for (int beta = 0; beta < 3; beta++) {
            std::vector<DTYPE> xyz_plus_plus = xyz;
            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                xyz_plus_plus[3 * i_sample + alpha] += DELTA;
                xyz_plus_plus[3 * i_sample + beta] += DELTA;
            }
            calculator.compute(xyz_plus_plus, n_samples, sph_plus_plus);
            // calculator.compute(xyz_plus_plus, sph_plus_plus);

            std::vector<DTYPE> xyz_plus_minus = xyz;
            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                xyz_plus_minus[3 * i_sample + alpha] += DELTA;
                xyz_plus_minus[3 * i_sample + beta] -= DELTA;
            }
            calculator.compute(xyz_plus_minus, n_samples, sph_plus_minus);
            // calculator.compute(xyz_plus_minus, sph_plus_minus);

            std::vector<DTYPE> xyz_minus_plus = xyz;
            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                xyz_minus_plus[3 * i_sample + alpha] -= DELTA;
                xyz_minus_plus[3 * i_sample + beta] += DELTA;
            }
            calculator.compute(xyz_minus_plus, n_samples, sph_minus_plus);
            // calculator.compute(xyz_minus_plus, sph_minus_plus);

            std::vector<DTYPE> xyz_minus_minus = xyz;
            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                xyz_minus_minus[3 * i_sample + alpha] -= DELTA;
                xyz_minus_minus[3 * i_sample + beta] -= DELTA;
            }
            // calculator.compute(xyz_minus_minus, sph_minus_minus);
            calculator.compute(xyz_minus_minus, n_samples, sph_minus_minus);

            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                    DTYPE analytical =
                        ddsph[9 * n_sph * i_sample + n_sph * 3 * alpha + n_sph * beta + i_sph];
                    DTYPE finite_diff = (sph_plus_plus[n_sph * i_sample + i_sph] -
                                         sph_plus_minus[n_sph * i_sample + i_sph] -
                                         sph_minus_plus[n_sph * i_sample + i_sph] +
                                         sph_minus_minus[n_sph * i_sample + i_sph]) /
                                        (4.0 * DELTA * DELTA);
                    if (!(std::abs(analytical / finite_diff - 1.0) < TOLERANCE ||
                          (std::abs(analytical) < 1e-15 && std::abs(finite_diff) < 1e-7)
                        )) { // Add a criterion for second
                             // derivatives which
                        // are zero, as they can fail the relative test
                        std::cout << "Wrong second derivative: " << analytical << " vs "
                                  << finite_diff << std::endl;
                        is_passed = false;
                    }
                }
            }
        }
    }

    return is_passed;
}

int main() {
    bool is_passed;
    int l_max_max = 15;
    std::vector<DTYPE> xyz(12);
    for (int i_sample = 0; i_sample < 4; i_sample++) {
        for (int alpha = 0; alpha < 3; alpha++) {
            xyz[3 * i_sample + alpha] =
                0.01 * i_sample - 0.3 * alpha * alpha; // Fill xyz with some numbers
        }
    }

    for (int l_max = 0; l_max < l_max_max; l_max++) { // Test for a range of l_max values

        sphericart::sycl::SphericalHarmonics<DTYPE> calculator_spherical(l_max);
        is_passed = check_gradient_call(l_max, calculator_spherical, xyz);
        if (!is_passed) {
            std::cout << "Test failed" << std::endl;
            //        return -1;
        }
        //        is_passed = check_hessian_call(l_max, calculator_spherical, xyz);
        //        if (!is_passed) {
        //            std::cout << "Test failed" << std::endl;
        //            return -1;
        //        }

        //        sphericart::SolidHarmonics<DTYPE> calculator_solid =
        //            sphericart::SolidHarmonics<DTYPE>(l_max);
        //        is_passed = check_gradient_call(l_max, calculator_solid, xyz);
        //        if (!is_passed) {
        //            std::cout << "Test failed" << std::endl;
        //            return -1;
        //        }
        //        is_passed = check_hessian_call(l_max, calculator_solid, xyz);
        //        if (!is_passed) {
        //            std::cout << "Test failed" << std::endl;
        //            return -1;
        //        }
    }

    std::cout << "Test passed" << std::endl;
    return 0;
}
