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

/**  
 * Opaque type to hold a spherical harmonics calculator object, that contains
 * pre-computed factors and allocated buffer space for calculations.
*/ 
struct sphericart_spherical_harmonics; 
typedef struct sphericart_spherical_harmonics sphericart_spherical_harmonics;

/**
 * Initialize a spherical harmonics calculator and returns a pointer 
 * that can then be used by functions that evaluate spherical harmonics
 * over arrays or individual samples.
 * 
 *  @param l_max
 *      The maximum degree of the spherical harmonics to be calculated.
 *  @param normalized
 *      If `false` (default) computes the scaled spherical harmonics, which are 
 *      polynomials in the Cartesian coordinates of the input points. If `true`,
 *      computes the normalized (spherical) spherical harmonics that are evaluated
 *      on the unit sphere. In practice, this simply computes the scaled harmonics
 *      at the normalized coordinates \f$(x/r, y/r, z/r)\f$, and adapts the derivatives
 *      accordingly. 
 * 
 * @return
 *      A pointer to a `sphericart_spherical_harmonics` object
 * 
*/
SPHERICART_EXPORT sphericart_spherical_harmonics *sphericart_new(size_t l_max, char normalized);
SPHERICART_EXPORT sphericart_spherical_harmonics *sphericart_new_f(size_t l_max, char normalized);

/**
 * Deletes a previously-allocated spherical harmonics calculator. 
*/
SPHERICART_EXPORT void sphericart_delete(sphericart_spherical_harmonics* spherical_harmonics);

/**
 * This function calculates the spherical harmonics and, optionally, their derivatives 
 * for an array of 3D points.
 *
 * @param spherical_harmonics
 *        A pointer to a `sphericart_spherical_harmonics` struct that holds prefactors
 *        and options to compute the spherical harmonics. 
 * @param n_samples 
 *        The number of 3D points for which the spherical harmonics will be calculated.
 * @param xyz 
 *        An array of size `(n_samples)*3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are to
 *        be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param sph 
 *        On entry, a (possibly uninitialized) array of size
 *        `n_samples*(l_max+1)*(l_max+1)`. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different
 *        samples, while the inner dimension is `(l_max+1)*(l_max+1)` long and
 *        it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain
 *        `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0),
 *        (2, 1), (2, 2)`, in this order.
 * @param dsph 
 *        On entry, either `NULL` or a (possibly uninitialized) array of
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
SPHERICART_EXPORT void sphericart_compute_array(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const double* xyz, double* sph, double* dsph);
SPHERICART_EXPORT void sphericart_compute_sample(sphericart_spherical_harmonics* spherical_harmonics, const double* xyz, double* sph, double* dsph);
SPHERICART_EXPORT void sphericart_compute_array_f(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const float* xyz, float* sph, float* dsph);
SPHERICART_EXPORT void sphericart_compute_sample_f(sphericart_spherical_harmonics* spherical_harmonics, const float* xyz, float* sph, float* dsph);

#ifdef __cplusplus
}
#endif

#endif
