/** @file test_derivatives.cpp
 *  @brief Tests derivatives using finite differences
*/

#include <iostream>
#include <vector>
#include <cassert>
#include "sphericart.hpp"

#define delta 1e-3
#define tolerance 1e-7


bool check_first_derivatives(int l_max, sphericart::SphericalHarmonics<double> &calculator, const std::vector<double> &xyz) {

    bool is_passed = true;
    int n_samples = xyz.size()/3;
    int n_sph = (l_max+1)*(l_max+1);

    std::vector<double> sph_plus;
    std::vector<double> sph_minus;
    std::vector<double> sph;
    std::vector<double> dsph;

    calculator.compute_with_gradients(xyz, sph, dsph);

    for (int alpha=0; alpha<3; alpha++) {
        std::vector<double> xyz_plus = xyz;
        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            xyz_plus[3*i_sample+alpha] += delta;
        }
        calculator.compute(xyz_plus, sph_plus);

        std::vector<double> xyz_minus = xyz;
        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            xyz_minus[3*i_sample+alpha] -= delta;
        }
        calculator.compute(xyz_minus, sph_minus);

        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            for (int i_sph=0; i_sph<n_sph; i_sph++) {
                double analytical = dsph[3*n_sph*i_sample+n_sph*alpha+i_sph];
                double finite_diff = (sph_plus[n_sph*i_sample+i_sph]-sph_minus[n_sph*i_sample+i_sph])/(2.0*delta);
                if (std::abs(analytical/finite_diff-1.0) > tolerance) std::cout << "Wrong first derivative: " << analytical << " vs " << finite_diff << std::endl;
                if (std::abs(analytical/finite_diff-1.0) > tolerance) is_passed = false;
            }
        }

    }

    return is_passed;
}


bool check_second_derivatives(int l_max, sphericart::SphericalHarmonics<double> &calculator, const std::vector<double> &xyz) {

    bool is_passed = true;
    int n_samples = xyz.size()/3;
    int n_sph = (l_max+1)*(l_max+1);

    std::vector<double> sph_plus_plus;
    std::vector<double> sph_plus_minus;
    std::vector<double> sph_minus_plus;
    std::vector<double> sph_minus_minus;
    std::vector<double> sph;
    std::vector<double> dsph;
    std::vector<double> ddsph;

    calculator.compute_with_hessians(xyz, sph, dsph, ddsph);

    for (int alpha=0; alpha<3; alpha++) {
        for (int beta=0; beta<3; beta++) {
            std::vector<double> xyz_plus_plus = xyz;
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_plus_plus[3*i_sample+alpha] += delta;
                xyz_plus_plus[3*i_sample+beta] += delta;
            }
            calculator.compute(xyz_plus_plus, sph_plus_plus);

            std::vector<double> xyz_plus_minus = xyz;
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_plus_minus[3*i_sample+alpha] += delta;
                xyz_plus_minus[3*i_sample+beta] -= delta;
            }
            calculator.compute(xyz_plus_minus, sph_plus_minus);

            std::vector<double> xyz_minus_plus = xyz;
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_minus_plus[3*i_sample+alpha] -= delta;
                xyz_minus_plus[3*i_sample+beta] += delta;
            }
            calculator.compute(xyz_minus_plus, sph_minus_plus);

            std::vector<double> xyz_minus_minus = xyz;
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_minus_minus[3*i_sample+alpha] -= delta;
                xyz_minus_minus[3*i_sample+beta] -= delta;
            }
            calculator.compute(xyz_minus_minus, sph_minus_minus);

            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                for (int i_sph=0; i_sph<n_sph; i_sph++) {
                    double analytical = ddsph[9*n_sph*i_sample+n_sph*3*alpha+n_sph*beta+i_sph];
                    double finite_diff = (sph_plus_plus[n_sph*i_sample+i_sph]-sph_plus_minus[n_sph*i_sample+i_sph]-sph_minus_plus[n_sph*i_sample+i_sph]+sph_minus_minus[n_sph*i_sample+i_sph])/(4.0*delta*delta);
                    std::cout << i_sph << " " << alpha << " " << beta << " " << finite_diff << " " << analytical << std::endl;
                    // if (std::abs(analytical/finite_diff-1.0) > tolerance) std::cout << "Wrong second derivative: " << analytical << " vs " << finite_diff << std::endl;
                    if (std::abs(analytical/finite_diff-1.0) > tolerance) is_passed = false;
                }
            }
        }

    }

    return is_passed;
}


int main() {

    bool is_passed;
    int l_max = 2;
    std::vector<double> xyz(15);
    for (int i_sample=0; i_sample<5; i_sample++) {
        for (int alpha=0; alpha<3; alpha++) {
            xyz[3*i_sample+alpha] = 0.01*i_sample - 0.3*alpha*alpha;  // Fill xyz with some numbers
        }
    }

    // Test without normalization:
    sphericart::SphericalHarmonics<double> calculator = sphericart::SphericalHarmonics<double>(l_max);
    /*
    is_passed = check_first_derivatives(l_max, calculator, xyz);
    if (!is_passed) {
        std::cout << "Test failed" << std::endl;
        return -1;
    }
    */
    is_passed = check_second_derivatives(l_max, calculator, xyz);
    if (!is_passed) {
        std::cout << "Test failed" << std::endl;
        return -1;
    }

    /*
    // Test with normalization:
    sphericart::SphericalHarmonics<double> normalized_calculator = sphericart::SphericalHarmonics<double>(l_max, true);
    is_passed = check_first_derivatives(l_max, normalized_calculator, xyz);
    if (!is_passed) {
        std::cout << "Test failed" << std::endl;
        return -1;
    }
    is_passed = check_second_derivatives(l_max, normalized_calculator, xyz);
    if (!is_passed) {
        std::cout << "Test failed" << std::endl;
        return -1;
    }
    */

    std::cout << "Test passed" << std::endl;
    return 0;
}
