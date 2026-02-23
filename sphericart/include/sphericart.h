/** \file sphericart.h
 *  Defines the C API for `sphericart`. Similar types and functions are
 *  available whether one is using the `double` or `float` data type, and whether
 *  one is using spherical or solid harmonics calculators.
 *  Types and functions for the `float` data type contain `_f` in their name,
 *  while those for the `double` data type do not.
 *  Similarly, types and functions for spherical harmonics calculations contain
 *  `_spherical_harmonics` in their name, and they calculate the
 *  real spherical harmonics \f$ Y^m_l \f$ as defined on Wikipedia,
 *  which are homogeneous polynomials of (x/r, y/r, z/r). In contrast, types
 *  and functions for solid harmonics calculations contain `_solid_harmonics` in
 *  their name, and thay calculate the same polynomials but as a function of the
 *  Cartesian coordinates (x, y, z), or, equivalently, \f$ r^l\,Y^m_l \f$.
 */

#ifndef SPHERICART_H
#define SPHERICART_H

#include "sphericart/exports.h"
#include "stdbool.h"
#include "stddef.h"

#ifdef __cplusplus

#include "sphericart.hpp"

using sphericart_spherical_harmonics_calculator_t = sphericart::SphericalHarmonics<double>;
using sphericart_spherical_harmonics_calculator_f_t = sphericart::SphericalHarmonics<float>;
using sphericart_solid_harmonics_calculator_t = sphericart::SolidHarmonics<double>;
using sphericart_solid_harmonics_calculator_f_t = sphericart::SolidHarmonics<float>;

extern "C" {

#else
/**
 * Opaque type to hold a spherical harmonics calculator object, that contains
 * pre-computed factors and allocated buffer space for calculations.
 *
 * The `sphericart_spherical_harmonics_calculator_t` performs calculations with `double` data type.
 */
struct sphericart_spherical_harmonics_calculator_t;

/**
 * A type referring to the `sphericart_spherical_harmonics_calculator_t` struct.
 */
typedef struct sphericart_spherical_harmonics_calculator_t sphericart_spherical_harmonics_calculator_t;

/**
 * Similar to `sphericart_spherical_harmonics_calculator_t`, but operating on the `float` data
 * type.
 */
struct sphericart_spherical_harmonics_calculator_f_t;

/**
 * A type referring to the `sphericart_spherical_harmonics_calculator_f_t` struct.
 */
typedef struct sphericart_spherical_harmonics_calculator_f_t
    sphericart_spherical_harmonics_calculator_f_t;

/**
 * Opaque type to hold a solid harmonics calculator object, that contains
 * pre-computed factors and allocated buffer space for calculations.
 *
 * The `sphericart_solid_harmonics_calculator_t` performs calculations with `double` data type.
 */
struct sphericart_solid_harmonics_calculator_t;

/**
 * A type referring to the `sphericart_solid_harmonics_calculator_t` struct.
 */
typedef struct sphericart_solid_harmonics_calculator_t sphericart_solid_harmonics_calculator_t;

/**
 * Similar to `sphericart_solid_harmonics_calculator_t`, but operating on the `float` data
 * type.
 */
struct sphericart_solid_harmonics_calculator_f_t;

/**
 * A type referring to the `sphericart_solid_harmonics_calculator_f_t` struct.
 */
typedef struct sphericart_solid_harmonics_calculator_f_t sphericart_solid_harmonics_calculator_f_t;
#endif

/**
 * Initializes a spherical harmonics calculator and returns a pointer that
 * can then be used by functions that evaluate spherical harmonics over
 * arrays or individual samples.
 *
 *  @param l_max The maximum degree of the spherical harmonics to be
 * calculated.
 *
 *  @return A pointer to a `sphericart_spherical_harmonics_calculator_t` object
 *
 */
SPHERICART_EXPORT sphericart_spherical_harmonics_calculator_t* sphericart_spherical_harmonics_new(
    size_t l_max
);

/**
 * Similar to `sphericart_spherical_harmonics_new`, but it returns a
 * `sphericart_spherical_harmonics_calculator_f_t`, which performs calculations on the `float` type.
 */
SPHERICART_EXPORT sphericart_spherical_harmonics_calculator_f_t* sphericart_spherical_harmonics_new_f(
    size_t l_max
);

/**
 * Deletes a previously allocated `sphericart_spherical_harmonics_calculator_t` calculator.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_delete(
    sphericart_spherical_harmonics_calculator_t* calculator
);

/**
 * Deletes a previously allocated `sphericart_spherical_harmonics_calculator_f_t` calculator.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_delete_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator
);

/**
 * This function calculates the spherical harmonics for an array of 3D points.
 *
 * @param calculator A pointer to a `sphericart_spherical_harmonics_calculator_t` struct
 *        that holds prefactors and options to compute the spherical
 *        harmonics.
 * @param xyz An array of size `n_samples x 3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are
 *        to be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param xyz_length size of the xyz allocation, i.e, `3 x n_samples`
 * @param sph pointer to the first element of an array containing `n_samples
 *        x (l_max + 1)^2` elements. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different
 *        samples, while the inner dimension size is `(l_max + 1)^2`
 *        long and it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain
 *        `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2,
 *        1), (2, 2)`, in this order.
 * @param sph_length size of the sph allocation, should be `n_samples x
 *        (l_max + 1)^2`
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * This function calculates the spherical harmonics and their
 * derivatives for an array of 3D points.
 *
 * @param calculator A pointer to a `sphericart_spherical_harmonics_calculator_t` struct
 *        that holds prefactors and options to compute the spherical
 *        harmonics.
 * @param xyz An array of size `n_samples x 3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are
 *        to be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param xyz_length size of the xyz allocation, i.e, `3 x n_samples``
 * @param sph pointer to the first element of an array containing `n_samples
 *        x (l_max + 1)^2` elements. On exit, this array will contain
 *        the spherical harmonics organized along two dimensions. The leading
 *        dimension is `n_samples` long and it represents the different
 *        samples, while the inner dimension size is ``(l_max + 1)^2``
 *        long and it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain
 *        `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2,
 *        1), (2, 2)`, in this order.
 * @param sph_length size of the sph allocation, should be `n_samples *
 *        (l_max + 1)^2`
 * @param dsph pointer to the first element of an array containing `n_samples
 *         x `n_samples x 3 x (l_max + 1)^2` elements. On exit, this
 *         array will contain the spherical harmonics' derivatives organized
 *         along three dimensions. As for the `sph` parameter, the leading
 *         dimension represents the different samples, while the inner-most
 *         dimension size is `(l_max + 1)^2`, and it represents
 *         the degree and order of the spherical harmonics (again, organized in
 *         lexicographic order). The intermediate dimension corresponds to
 *         different spatial derivatives of the spherical harmonics: x, y,
 *         and z, respectively.
 * @param dsph_length size of the dsph allocation, which should be `n_samples
 *         x 3 x (l_max + 1)^2`
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array_with_gradients(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * This function calculates the spherical harmonics, their
 * derivatives and second derivatives for an array of 3D points.
 *
 * @param calculator A pointer to a `sphericart_spherical_harmonics_calculator_t` struct
 *        that holds prefactors and options to compute the spherical
 *        harmonics.
 * @param xyz An array of size `n_samples x 3`. It contains the Cartesian
 *        coordinates of the 3D points for which the spherical harmonics are
 *        to be computed, organized along two dimensions. The outer dimension is
 *        `n_samples` long, accounting for different samples, while the inner
 *        dimension has size 3 and it represents the x, y, and z coordinates
 *        respectively.
 * @param xyz_length size of the xyz allocation, i.e, ``3 x n_samples``
 * @param sph pointer to the first element of an array containing
 *        `n_samples x (l_max + 1)^2` elements. On exit, this array
 *        will contain the spherical harmonics organized along two dimensions.
 *        The leading dimension is `n_samples` long and it represents the different
 *        samples, while the inner dimension size is `(l_max + 1)^2`
 *        long and it contains the spherical harmonics. These are laid out in
 *        lexicographic order. For example, if `l_max=2`, it will contain
 *        ``(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2,
 *        1), (2, 2)``, in this order.
 * @param sph_length size of the sph allocation, should be `n_samples *
 *        (l_max + 1)^2`
 * @param dsph pointer to the first element of an array containing
 *         `n_samples x 3 x (l_max + 1)^2` elements. On exit,
 *         this array will contain the spherical harmonics' derivatives organized
 *         along three dimensions. As for the `sph` parameter, the leading
 *         dimension represents the different samples, while the inner-most
 *         dimension size is `(l_max + 1)^2`, and it represents
 *         the degree and order of the spherical harmonics (again, organized in
 *         lexicographic order). The intermediate dimension corresponds to
 *         different spatial derivatives of the spherical harmonics: x, y,
 *         and z, respectively.
 * @param dsph_length size of the dsph allocation, which should be `n_samples
 *         x 3 x (l_max + 1)^2`
 * @param ddsph pointer to the first element of an array containing
 *        `n_samples x 3 x 3 x (l_max + 1)^2` elements. On exit,
 *         this array will contain the spherical harmonics' second derivatives
 *         organized along four dimensions. As for the `sph` parameter, the leading
 *         dimension represents the different samples, while the inner-most dimension
 *         size is `(l_max + 1)^2`, and it represents the degree and
 *         order of the spherical harmonics (again, organized in lexicographic
 *         order). The intermediate dimensions correspond to the different spatial
 *         second derivatives of the spherical harmonics, i.e., to the dimensions of
 *         the hessian matrix.
 * @param ddsph_length size of the dsph allocation, which should be
 *         `n_samples x 3 x 3* (l_max + 1)^2`
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array_with_hessians(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array`, but it computes the spherical
 * harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array_with_gradients`, but it
 * computes the spherical harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample_with_gradients(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array_with_hessians`, but it computes
 * the spherical harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample_with_hessians(
    sphericart_spherical_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array`, but using the `float` data
 * type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array_with_gradients`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array_with_gradients_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array_with_hessians`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_array_with_hessians_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
);

/**
 * Get the number of OpenMP threads used by a calculator.
 * If `sphericart` is computed without OpenMP support returns 1.
 */
SPHERICART_EXPORT int sphericart_spherical_harmonics_omp_num_threads(
    sphericart_spherical_harmonics_calculator_t* calculator
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_omp_num_threads`, but for
 * a `float` calculator.
 */
SPHERICART_EXPORT int sphericart_spherical_harmonics_omp_num_threads_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_sample`, but using the `float` data
 * type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_sample_with_gradients`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample_with_gradients_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_sample_with_hessians`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_spherical_harmonics_compute_sample_with_hessians_f(
    sphericart_spherical_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_new`, but it returns a
 * `sphericart_solid_harmonics_calculator_t`, which perform solid harmonics calculations.
 */
SPHERICART_EXPORT sphericart_solid_harmonics_calculator_t* sphericart_solid_harmonics_new(size_t l_max
);

/**
 * Similar to `sphericart_solid_harmonics_new`, but it returns a
 * `sphericart_solid_harmonics_calculator_f_t`, which performs calculations on the `float` type.
 */
SPHERICART_EXPORT sphericart_solid_harmonics_calculator_f_t* sphericart_solid_harmonics_new_f(size_t l_max
);

/**
 * Deletes a previously allocated `sphericart_solid_harmonics_calculator_t` calculator.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_delete(
    sphericart_solid_harmonics_calculator_t* calculator
);

/**
 * Deletes a previously allocated `sphericart_solid_harmonics_calculator_f_t` calculator.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_delete_f(
    sphericart_solid_harmonics_calculator_f_t* calculator
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_compute_array`, but it computes the solid
 * harmonics.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_gradients`, but it computes the
 * solid harmonics and their derivatives.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array_with_gradients(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_hessians`, but it computes the
 * solid harmonics and their derivatives.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array_with_hessians(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array`, but it computes the solid
 * harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_gradients`, but it
 * computes the solid harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample_with_gradients(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_hessians`, but it computes
 * the solid harmonics for a single 3D point in space.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample_with_hessians(
    sphericart_solid_harmonics_calculator_t* calculator,
    const double* xyz,
    size_t xyz_length,
    double* sph,
    size_t sph_length,
    double* dsph,
    size_t dsph_length,
    double* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array`, but using the `float` data
 * type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_gradients`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array_with_gradients_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_array_with_hessians`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_array_with_hessians_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_sample`, but using the `float` data
 * type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_sample_with_gradients`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample_with_gradients_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length
);

/**
 * Similar to :func:`sphericart_solid_harmonics_compute_sample_with_hessians`, but using the
 * `float` data type.
 */
SPHERICART_EXPORT void sphericart_solid_harmonics_compute_sample_with_hessians_f(
    sphericart_solid_harmonics_calculator_f_t* calculator,
    const float* xyz,
    size_t xyz_length,
    float* sph,
    size_t sph_length,
    float* dsph,
    size_t dsph_length,
    float* ddsph,
    size_t ddsph_length
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_omp_num_threads`, but for a solid harmonics
 * calculator.
 */
SPHERICART_EXPORT int sphericart_solid_harmonics_omp_num_threads(
    sphericart_solid_harmonics_calculator_t* calculator
);

/**
 * Similar to :func:`sphericart_spherical_harmonics_omp_num_threads`, but for a `float` solid
 * harmonics calculator.
 */
SPHERICART_EXPORT int sphericart_solid_harmonics_omp_num_threads_f(
    sphericart_solid_harmonics_calculator_f_t* calculator
);

#ifdef __cplusplus
}
#endif

#endif
