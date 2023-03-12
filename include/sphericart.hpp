/** \file sphericart.hpp
* The C++ API for `sphericart` contains a few compiled functions. 
* More fine-grained control over the behavior of the functions can be 
* achieved using the templates in `template.hpp`
*/

#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include <cstddef>

#include "sphericart/exports.h"

namespace sphericart {

/**
 * Holds prefactors for evaluating \f$Y_l^m\f$ and \f$Q_l^m\f$.
 * 
 * @var yq_buffer::y 
 *      The prefactors for \f$Y_l^m\f$ 
 * @var yq_buffer::q 
 *      Coefficients used to evaluate \f$Q_l^m\f$
*/
template <typename DTYPE>
struct yq_buffer {
    DTYPE y, q; 
};


/**
 * Used to store temporary values corresponding to the (scaled) 
 * \f$\cos m \phi\f$, \f$\sin m \phi\f$, and \f$m z\f$.
 *
 * @var csz_buffer::c
 *      Holds the value of the scaled \f$\sin m \phi\f$
 * @var csz_buffer::s
 *      Holds the value of the scaled \f$\cos m \phi\f$
 * @var csz_buffer::z
 *      Holds the values of \f$2(m+1)z\f$
 * 
*/
template <typename DTYPE>
struct csz_buffer {
    DTYPE c, s, z;
};

/**
 * This function calculates the prefactors needed for the computation of the
 * spherical harmonics.
 *
 * @param l_max The maximum degree of spherical harmonics for which the
 *        prefactors will be calculated.
 * @param factors On entry, a (possibly uninitialized) array of size
 *        `(l_max+1) * (l_max+2)/2`. On exit, it will contain the prefactors for
 *        the calculation of the spherical harmonics up to degree `l_max`, in
 *        the order `(l, m) = (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), ...`. 
 *        Each entry contains the numerical prefactors that enter the full \f$Y_l^m\f$,
 *        and the constansts that are needed to evaluate the \f$Q_l^m\f$.
 */
void SPHERICART_EXPORT compute_sph_prefactors(int l_max, yq_buffer<double> *factors);
// void SPHERICART_EXPORT compute_sph_prefactors(int l_max, float *factors);

/**
 * This function calculates the spherical harmonics and, optionally, their
 * derivatives for a set of 3D points.
 *
 * @param n_samples The number of 3D points for which the spherical harmonics
 *        will be calculated.
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the
 *        compute_sph_prefactors() function.
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
    const yq_buffer<double>* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);
/*
void SPHERICART_EXPORT cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const float* prefactors,
    const float *xyz,
    float *sph,
    float *dsph
);
*/

/**
 * This function calculates the conventional (normalized) spherical harmonics and, 
 * optionally, their derivatives for a set of 3D points. 
 * Takes the same arguments as cartesian_spherical_harmonics(), and simply returns
 * values evaluated for the normalized positions \f$(x/r, y/r, z/r)\f$, with the corresponding
 * derivatives.
 */
void SPHERICART_EXPORT normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const yq_buffer<double>* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);
/*
void SPHERICART_EXPORT normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const float* prefactors,
    const float *xyz,
    float *sph,
    float *dsph
);
*/

/**
 * This function calculates the spherical harmonics and, optionally, their
 * derivatives for a single 3D point. For a more flexible (and slightly faster) 
 * implementation, you may want to look at the template functions defined 
 * in `templates.hpp`. See also cartesian_spherical_harmonics() for a more 
 * thorough discussion of the parameters. 
 *
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the
 *        compute_sph_prefactors() function.
 * @param buffer A buffer of size `(l_max+1)*(l_max+2)/2)` that will be used to 
 *        store the sines, cosines and m*z values for the point
 * @param xyz_i An array of size 3 containing the Cartesian coordinates of the 
 *        selected point. 
 * @param sph_i On entry, a (possibly uninitialized) array of size
 *        `(l_max+1)*(l_max+1)`.
 * @param dsph_i On entry, a (possibly uninitialized) array of
 *        size `3*(l_max+1)*(l_max+1)`. On exit, this array will contain the 
 *        spherical harmonics' derivatives organized along three dimensions. 
 */
void cartesian_spherical_harmonics_sample(int l_max,
    const yq_buffer<double>* prefactors,
    csz_buffer<double>* buffer,
    const double *xyz_i,
    double *sph_i,
    double *dsph_i
);
} //namespace sphericart

#endif
