/** \file sphericart.h
*  Defines the C API for `sphericart`.
*/

#ifndef SPHERICART_H
#define SPHERICART_H

#include "sphericart/exports.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sphericart_spherical_harmonics; 
typedef struct sphericart_spherical_harmonics sphericart_spherical_harmonics;
SPHERICART_EXPORT sphericart_spherical_harmonics * sphericart_new(size_t l_max, char normalized);
SPHERICART_EXPORT sphericart_spherical_harmonics *  sphericart_new_f(size_t l_max, char normalized);
SPHERICART_EXPORT void sphericart_delete(sphericart_spherical_harmonics* spherical_harmonics);
SPHERICART_EXPORT void sphericart_compute_array(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const double* xyz, double* sph, double* dsph);
SPHERICART_EXPORT void sphericart_compute_sample(sphericart_spherical_harmonics* spherical_harmonics, const double* xyz, double* sph, double* dsph);
SPHERICART_EXPORT void sphericart_compute_array_f(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const float* xyz, float* sph, float* dsph);
SPHERICART_EXPORT void sphericart_compute_sample_f(sphericart_spherical_harmonics* spherical_harmonics, const float* xyz, float* sph, float* dsph);

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
 *        the first holds the numerical prefactors that enter the full \f$Y_l^m\f$,
 *        the second containing constansts that are needed to evaluate the \f$Q_l^m\f$.
 */
void SPHERICART_EXPORT sphericart_compute_sph_prefactors(int l_max, double *factors);

/**
 * This function calculates the Cartesian (un-normalized) spherical harmonics and, 
 * optionally, their derivatives for a set of 3D points.
 *
 * @param n_samples The number of 3D points for which the spherical harmonics
 *        will be calculated.
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the
 *        sphericart_compute_sph_prefactors() function.
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
void SPHERICART_EXPORT sphericart_cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);

/**
 * This function calculates the conventional (normalized) spherical harmonics and, 
 * optionally, their derivatives for a set of 3D points. 
 * Takes the same arguments as sphericart_cartesian_spherical_harmonics(), and simply returns
 * values evaluated for the normalized positions \f$(x/r, y/r, z/r)\f$, with the corresponding
 * derivatives.
 */
void SPHERICART_EXPORT sphericart_normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);

#ifdef __cplusplus
}
#endif

#endif
