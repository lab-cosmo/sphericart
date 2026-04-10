/** @file test_derivatives.cpp
 *  @brief Tests derivatives using finite differences
 */

#include <cassert>
#include <iostream>
#include <vector>

#include "sphericart_sycl.hpp"
#include "sycl_base.hpp"

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
bool check_hessian_call(int l_max, C<DTYPE>& calculator, const std::vector<DTYPE>& xyz_host) {
    bool is_passed = true;
    int n_samples = xyz_host.size() / 3;
    int n_sph = (l_max + 1) * (l_max + 1);

    DEVICE_INIT(DTYPE, xyz, xyz_host.data(), xyz_host.size());
    MALLOC(DTYPE, sph, n_samples * n_sph);
    MALLOC(DTYPE, dsph, 3 * n_samples * n_sph);
    MALLOC(DTYPE, ddsph, 9 * n_samples * n_sph);

    calculator.compute_with_hessians(xyz, n_samples, sph, dsph, ddsph);

    auto dsph_host = std::vector<DTYPE>(3 * n_samples * n_sph, 0.0);
    auto ddsph_host = std::vector<DTYPE>(9 * n_samples * n_sph, 0.0);
    DEVICE_GET(DTYPE, dsph_host.data(), dsph, dsph_host.size());
    DEVICE_GET(DTYPE, ddsph_host.data(), ddsph, ddsph_host.size());
    FREE(sph);
    FREE(dsph);
    FREE(ddsph);
    FREE(xyz);

    // Check first derivatives via finite differences:
    for (int alpha = 0; alpha < 3; alpha++) {
        auto sph_plus_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);
        auto sph_minus_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);

        MALLOC(DTYPE, sph_plus, n_samples * n_sph);
        MALLOC(DTYPE, sph_minus, n_samples * n_sph);

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
                if (std::abs(analytical / finite_diff - 1.0) > TOLERANCE) {
                    std::cout << "Wrong first derivative: " << analytical << " vs " << finite_diff
                              << std::endl;
                    is_passed = false;
                }
            }
        }
        FREE(sph_plus);
        FREE(sph_minus);
    }

    // Check second derivatives via finite differences:
    for (int alpha = 0; alpha < 3; alpha++) {
        for (int beta = 0; beta < 3; beta++) {
            auto sph_pp_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);
            auto sph_pm_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);
            auto sph_mp_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);
            auto sph_mm_host = std::vector<DTYPE>(n_samples * n_sph, 0.0);

            auto compute_perturbed = [&](std::vector<DTYPE>& out, DTYPE da, DTYPE db) {
                std::vector<DTYPE> xyz_p = xyz_host;
                for (int i = 0; i < n_samples; i++) {
                    xyz_p[3 * i + alpha] += da;
                    xyz_p[3 * i + beta] += db;
                }
                DEVICE_INIT(DTYPE, xyz_dev, xyz_p.data(), xyz_p.size());
                MALLOC(DTYPE, sph_dev, n_samples * n_sph);
                calculator.compute(xyz_dev, n_samples, sph_dev);
                DEVICE_GET(DTYPE, out.data(), sph_dev, out.size());
                FREE(xyz_dev);
                FREE(sph_dev);
            };

            compute_perturbed(sph_pp_host, +DELTA, +DELTA);
            compute_perturbed(sph_pm_host, +DELTA, -DELTA);
            compute_perturbed(sph_mp_host, -DELTA, +DELTA);
            compute_perturbed(sph_mm_host, -DELTA, -DELTA);

            for (int i_sample = 0; i_sample < n_samples; i_sample++) {
                for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                    DTYPE analytical =
                        ddsph_host[9 * n_sph * i_sample + n_sph * 3 * alpha + n_sph * beta + i_sph];
                    DTYPE finite_diff = (sph_pp_host[n_sph * i_sample + i_sph] -
                                         sph_pm_host[n_sph * i_sample + i_sph] -
                                         sph_mp_host[n_sph * i_sample + i_sph] +
                                         sph_mm_host[n_sph * i_sample + i_sph]) /
                                        (4.0 * DELTA * DELTA);
                    if (!(std::abs(analytical / finite_diff - 1.0) < TOLERANCE ||
                          (std::abs(analytical) < 1e-15 && std::abs(finite_diff) < 1e-7))) {
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

bool check_backward_call(int l_max, const std::vector<DTYPE>& xyz_host) {
    bool is_passed = true;
    int n_samples = xyz_host.size() / 3;
    int n_sph = (l_max + 1) * (l_max + 1);

    // Compute dsph on device via the forward pass
    DEVICE_INIT(DTYPE, xyz, xyz_host.data(), xyz_host.size());
    MALLOC(DTYPE, sph, n_samples * n_sph);
    MALLOC(DTYPE, dsph, 3 * n_samples * n_sph);
    sphericart::sycl::SphericalHarmonics<DTYPE> calc(l_max);
    calc.compute_with_gradients(xyz, n_samples, sph, dsph);
    FREE(sph);
    FREE(xyz);

    // Use all-ones upstream gradient so the expected result is just the row-sum
    // of dsph along the harmonic index
    std::vector<DTYPE> sph_grad_host(n_samples * n_sph, 1.0);
    DEVICE_INIT(DTYPE, sph_grad, sph_grad_host.data(), sph_grad_host.size());

    MALLOC(DTYPE, xyz_grad, n_samples * 3);
    sphericart::sycl::spherical_harmonics_backward_sycl_base(
        dsph, sph_grad, n_samples, n_sph, xyz_grad
    );

    // Copy results back to host for comparison
    std::vector<DTYPE> xyz_grad_host(n_samples * 3, 0.0);
    DEVICE_GET(DTYPE, xyz_grad_host.data(), xyz_grad, xyz_grad_host.size());

    std::vector<DTYPE> dsph_host(3 * n_samples * n_sph, 0.0);
    DEVICE_GET(DTYPE, dsph_host.data(), dsph, dsph_host.size());

    // Reference: xyz_grad[i, alpha] = sum_j dsph[i, alpha, j] * sph_grad[i, j]
    for (int i = 0; i < n_samples; i++) {
        for (int alpha = 0; alpha < 3; alpha++) {
            DTYPE expected = 0.0;
            for (int j = 0; j < n_sph; j++) {
                expected +=
                    dsph_host[i * 3 * n_sph + alpha * n_sph + j] * sph_grad_host[i * n_sph + j];
            }
            DTYPE computed = xyz_grad_host[i * 3 + alpha];
            bool close;
            if (std::abs(expected) > 1e-10) {
                close = std::abs(computed / expected - 1.0) < TOLERANCE;
            } else {
                close = std::abs(computed - expected) < 1e-10;
            }
            if (!close) {
                std::cout << "Wrong backward gradient at sample=" << i << " alpha=" << alpha
                          << ": computed=" << computed << " expected=" << expected << std::endl;
                is_passed = false;
            }
        }
    }

    FREE(dsph);
    FREE(sph_grad);
    FREE(xyz_grad);

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
            std::cout << "Forward gradient test failed for l_max=" << l_max << std::endl;
            return 1;
        }

        is_passed = check_hessian_call(l_max, calculator_spherical, xyz);
        if (!is_passed) {
            std::cout << "Hessian test failed for l_max=" << l_max << std::endl;
            return 1;
        }

        is_passed = check_backward_call(l_max, xyz);
        if (!is_passed) {
            std::cout << "Backward test failed for l_max=" << l_max << std::endl;
            return 1;
        }
    }

    std::cout << "Test passed" << std::endl;
    return 0;
}
