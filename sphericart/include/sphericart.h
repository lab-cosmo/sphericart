/** \file sphericart.h
*  Defines the C API for `sphericart`.
*/

#ifndef SPHERICART_H
#define SPHERICART_H

#include "sphericart/exports.h"
#include "stddef.h"
#include "stdbool.h"

#ifdef __cplusplus

#include "sphericart.hpp"

using sphericart_calculator_t = sphericart::SphericalHarmonics<double>;
using sphericart_calculator_f_t = sphericart::SphericalHarmonics<float>;

extern "C" {

#else
/**
 * Opaque type to hold a spherical harmonics calculator object, that contains
 * pre-computed factors and allocated buffer space for calculations.
 *
 * The `sphericart_calculator_t` performs calculations with `double` data type.
*/
struct sphericart_calculator_t;

/**
 * A type referring to the `sphericart_calculator_t` struct.
*/
typedef struct sphericart_calculator_t sphericart_calculator_t;

/**
 * Similar to `sphericart_calculator_t`, but operating on the `float` data type.
*/
struct sphericart_calculator_f_t;

/**
 * A type referring to the `sphericart_calculator_f_t` struct.
*/
typedef struct sphericart_calculator_f_t sphericart_calculator_f_t;
#endif

/**
 * Initializes a spherical harmonics calculator and returns a pointer that can
 * then be used by functions that evaluate spherical harmonics over arrays or
 * individual samples.
 *
 *  @param l_max The maximum degree of the spherical harmonics to be calculated.
 *  @param normalized If `false`, computes the scaled spherical harmonics, which
 *      are polynomials in the Cartesian coordinates of the input points. If
 *      `true`, computes the normalized spherical harmonics that are
 *      evaluated on the unit sphere. In practice, this simply computes the
 *      scaled harmonics at the normalized coordinates \f$(x/r, y/r, z/r)\f$,
 *      and adapts the derivatives accordingly.
 *
 *  @return A pointer to a `sphericart_calculator_t` object
 *
*/
SPHERICART_EXPORT sphericart_calculator_t* sphericart_new(size_t l_max, bool normalized);

/**
 * Similar to `sphericart_new`, but it returns a `sphericart_calculator_f_t`, which
 * performs calculations on the `float` type.
*/
SPHERICART_EXPORT sphericart_calculator_f_t* sphericart_new_f(size_t l_max, bool normalized);

/**
 * Deletes a previously allocated `sphericart_calculator_t` calculator.
*/
SPHERICART_EXPORT void sphericart_delete(sphericart_calculator_t* calculator);

/**
 * Deletes a previously allocated `sphericart_calculator_f_t` calculator.
*/
SPHERICART_EXPORT void sphericart_delete_f(sphericart_calculator_f_t* calculator);

/**
 * This function calculates the spherical harmonics and, optionally, their
 * derivatives for an array of 3D points.
 *
 * @param spherical_harmonics A pointer to a `sphericart_calculator_t` struct
 *        that holds prefactors and options to compute the spherical harmonics.
 * @param xyz An array of size `n_samples x 3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are to
 *        be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param xyz_length size of the xyz allocation, i.e, `3 * n_samples`
 * @param sph pointer to the first element of an array containing `n_samples *
 *        (l_max + 1) * (l_max + 1)` elements. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different samples,
 *        while the inner dimension size is `(l_max + 1) * (l_max + 1)` long and
 *        it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain `(l,
 *        m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2,
 *        1), (2, 2)`, in this order.
 * @param sph_length size of the sph allocation, should be `n_samples * (l_max +
 *        1) * (l_max + 1)`
 */
SPHERICART_EXPORT void sphericart_compute_array(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * This function calculates the spherical harmonics and their
 * derivatives for an array of 3D points.
 *
 * @param spherical_harmonics A pointer to a `sphericart_calculator_t` struct
 *        that holds prefactors and options to compute the spherical harmonics.
 * @param xyz An array of size `n_samples x 3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are to
 *        be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param xyz_length size of the xyz allocation, i.e, `3 * n_samples`
 * @param sph pointer to the first element of an array containing `n_samples *
 *        (l_max + 1) * (l_max + 1)` elements. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different samples,
 *        while the inner dimension size is `(l_max + 1) * (l_max + 1)` long and
 *        it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain `(l,
 *        m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2,
 *        1), (2, 2)`, in this order.
 * @param sph_length size of the sph allocation, should be `n_samples * (l_max +
 *        1) * (l_max + 1)`
 * @param dsph pointer to the first element of an array containing `n_samples *
 *         `n_samples * 3 * (l_max + 1) * (l_max + 1)` elements. On exit, this
 *         array will contain the spherical harmonics' derivatives organized
 *         along three dimensions. As for the `sph` parameter, the leading
 *         dimension represents the different samples, while the inner-most
 *         dimension size is `(l_max + 1) * (l_max + 1)`, and it represents the
 *         degree and order of the spherical harmonics (again, organized in
 *         lexicographic order). The intermediate dimension corresponds to
 *         different spatial derivatives of the spherical harmonics: x, y, and
 *         z, respectively.
 * @param dsph_length size of the dsph allocation, which should be `n_samples * 3 *
 *        (l_max + 1) * (l_max + 1)`
 */
SPHERICART_EXPORT void sphericart_compute_array_with_gradients(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_compute_array`, but it computes the spherical
 * harmonics for a single 3D point in space.
*/
SPHERICART_EXPORT void sphericart_compute_sample(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_compute_array_with_gradients`, but it computes the
 * spherical harmonics for a single 3D point in space.
*/
SPHERICART_EXPORT void sphericart_compute_sample_with_gradients(
    sphericart_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_compute_array`, but using the `float` data type.
*/
SPHERICART_EXPORT void sphericart_compute_array_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_compute_array_with_gradients`, but using the
 * `float` data type.
*/
SPHERICART_EXPORT void sphericart_compute_array_with_gradients_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_compute_sample`, but using the `float` data type.
*/
SPHERICART_EXPORT void sphericart_compute_sample_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_compute_sample_with_gradients`, but using the `float` data type.
*/
SPHERICART_EXPORT void sphericart_compute_sample_with_gradients_f(
    sphericart_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Get the number of OpenMP threads used by a calculator.
 * If `sphericart` is computed without OpenMP support returns 1.
*/
SPHERICART_EXPORT int sphericart_omp_num_threads(sphericart_calculator_t* calculator);

SPHERICART_EXPORT int sphericart_omp_num_threads_f(sphericart_calculator_f_t* calculator);

#ifdef __cplusplus
}
#endif

#endif
