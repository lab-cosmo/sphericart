#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include <cstddef>

#include "sphericart/exports.h"

namespace sphericart {

/**
 * This function calculates the prefactors needed for the computation of the
 * spherical harmonics.
 *
 * @param l_max The maximum degree of spherical harmonics for which the
 *        prefactors will be calculated.
 * @param factors On entry, a (possibly uninitialized) array of size
 *        `(l_max+1) * (l_max+2)`. On exit, it will contain the prefactors for
 *        the calculation of the spherical harmonics up to degree `l_max`, in
 *        the order `(l, m) = (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), ...`. 
 *        The array contains two blocks of size `(l_max+1) * (l_max+2) / 2`: 
 *        the first holds the numerical prefactors that enter the full $Y_l^m$,
 *        the second containing constansts that are needed to evaluate the $Q_l^m$.
 */
void SPHERICART_EXPORT compute_sph_prefactors(int l_max, double *factors);
void SPHERICART_EXPORT compute_sph_prefactors(int l_max, float *factors);

/**
 * This function calculates the spherical harmonics and, optionally, their
 * derivatives for a set of 3D points.
 *
 * @param n_samples The number of 3D points for which the spherical harmonics
 *        will be calculated.
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the
 *        `compute_sph_prefactors` function.
 * @param xyz An array of size `(n_samples)*3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are to
 *        be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param sph On entry, a (possibly uninitialized) array of size
 *        `n_samples*(l_max+1)*(l_max+1)`. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different
 *        samples, while the inner dimension is `(l_max+1)*(l_max+1)` long and
 *        it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain
 *        `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0),
 *        (2, 1), (2, 2)`, in this order.
 * @param dsph On entry, either `NULL` or a (possibly uninitialized) array of
 *        size `n_samples*3*(l_max+1)*(l_max+1)`. If `dsph` is `NULL`, the
 *        spherical harmonics' derivatives will not be calculated. Otherwise, on
 *        exit, this array will contain the spherical harmonics' derivatives
 *        organized along three dimensions. As for the `sph` parameter, the
 *        leading dimension represents the different samples, while the
 *        inner-most dimension is `(l_max+1)*(l_max+1)`, and it represents the
 *        degree and order of the spherical harmonics (again, organized in
 *        lexicographic order). The intermediate dimension corresponds to
 *        different spatial derivatives of the spherical harmonics: x, y, and z,
 *        respectively.
 */
void SPHERICART_EXPORT cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);
void SPHERICART_EXPORT cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const float* prefactors,
    const float *xyz,
    float *sph,
    float *dsph
);

/**
 * This function calculates the conventional (normalized) spherical harmonics and, 
 * optionally, their derivatives for a set of 3D points. 
 * Takes the same arguments as `sphericart_spherical_harmonics`, and simply returns
 * values evaluated for the normalized positions (x/r, y/r, z/r), with the corresponding
 * derivatives.
 */
void SPHERICART_EXPORT normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);
void SPHERICART_EXPORT normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const float* prefactors,
    const float *xyz,
    float *sph,
    float *dsph
);

} // namespace sphericart

#endif
