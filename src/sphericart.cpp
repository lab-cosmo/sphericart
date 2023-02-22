#include <cmath>

#include "sphericart.hpp"
#include "templates.hpp"

#define _HC_LMAX 6

void sphericart::compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
        Use an iterative formula to avoid computing a ratio
        of factorials, and incorporates the 1/sqrt(2) that
        is associated with the Yl0's
    */

    unsigned int k = 0; // quick access index
    for (unsigned int l = 0; l <= l_max; ++l) {
        double factor = (2 * l + 1) / (2 * M_PI);
        // incorporates  the 1/sqrt(2) that goes with the m=0 SPH
        factors[k] = sqrt(factor) * M_SQRT1_2;
        for (int m = 1; m <= l; ++m) {
            factor *= 1.0 / (l * (l + 1) + m * (1 - m));
            if (m % 2 == 0) {
                factors[k + m] = sqrt(factor);
            } else {
                factors[k + m] = -sqrt(factor);
            }
        }
        k += l + 1;
    }
}

void cartesian_spherical_harmonics(
    unsigned int n_samples,
    unsigned int l_max,
    const double *prefactors,
    double *xyz,
    double *sph,
    double *dsph
) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the
        r^l term can be easily incorporated into any radial function or
        added a posteriori (with the corresponding derivative).
    */

    // call directly the fast ones
    if (l_max <= _HC_LMAX) {
        if (dsph == nullptr) {
            switch (l_max) {
            case 0:
                cartesian_spherical_harmonics_hc<false, 0>(n_samples, xyz, sph, dsph);
                break;
            case 1:
                cartesian_spherical_harmonics_hc<false, 1>(n_samples, xyz, sph, dsph);
                break;
            case 2:
                cartesian_spherical_harmonics_hc<false, 2>(n_samples, xyz, sph, dsph);
                break;
            case 3:
                cartesian_spherical_harmonics_hc<false, 3>(n_samples, xyz, sph, dsph);
                break;
            case 4:
                cartesian_spherical_harmonics_hc<false, 4>(n_samples, xyz, sph, dsph);
                break;
            case 5:
                cartesian_spherical_harmonics_hc<false, 5>(n_samples, xyz, sph, dsph);
                break;
            case 6:
                cartesian_spherical_harmonics_hc<false, 6>(n_samples, xyz, sph, dsph);
                break;
            }

        } else {
            switch (l_max) {
            case 0:
                cartesian_spherical_harmonics_hc<true, 0>(n_samples, xyz, sph, dsph);
                break;
            case 1:
                cartesian_spherical_harmonics_hc<true, 1>(n_samples, xyz, sph, dsph);
                break;
            case 2:
                cartesian_spherical_harmonics_hc<true, 2>(n_samples, xyz, sph, dsph);
                break;
            case 3:
                cartesian_spherical_harmonics_hc<true, 3>(n_samples, xyz, sph, dsph);
                break;
            case 4:
                cartesian_spherical_harmonics_hc<true, 4>(n_samples, xyz, sph, dsph);
                break;
            case 5:
                cartesian_spherical_harmonics_hc<true, 5>(n_samples, xyz, sph, dsph);
                break;
            case 6:
                cartesian_spherical_harmonics_hc<true, 6>(n_samples, xyz, sph, dsph);
                break;
            }
        }
    } else {
        if (dsph == nullptr) {
            _compute_sphcrt_templated<false, _HC_LMAX>(n_samples, l_max, prefactors, xyz, sph, dsph);
        } else {
            _compute_sphcrt_templated<true, _HC_LMAX>(n_samples, l_max, prefactors, xyz, sph, dsph);
        }
    }
}
