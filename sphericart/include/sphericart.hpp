/** \file sphericart.hpp
*  Defines the C++ API for `sphericart`.
*/

#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include <cstddef>
#include <vector>
#include <tuple>

#include "sphericart/exports.h"

#ifdef _SPHERICART_INTERNAL_IMPLEMENTATION
    #include "macros.hpp"
    #include "templates.hpp"
#endif


namespace sphericart {

/**
 * A spherical harmonics calculator.
 *
 * It handles initialization of the prefactors upon initialization and it stores the
 * buffers that are necessary to compute the spherical harmonics efficiently.
*/
template<typename T>
class SphericalHarmonics{
public:
    /** Initialize the SphericalHarmonics class setting maximum degree and normalization
     *
     *  @param l_max
     *      The maximum degree of the spherical harmonics to be calculated.
     *  @param normalized
     *      If `false` (default) computes the scaled spherical harmonics, which are
     *      polynomials in the Cartesian coordinates of the input points. If `true`,
     *      computes the normalized spherical harmonics that are evaluated
     *      on the unit sphere. In practice, this simply computes the scaled harmonics
     *      at the normalized coordinates \f$(x/r, y/r, z/r)\f$, and adapts the derivatives
     *      accordingly.
     */
    SphericalHarmonics(size_t l_max, bool normalized=false);

    /* @cond */
    ~SphericalHarmonics();
    /* @endcond */

    /** Computes the spherical harmonics for one or more 3D points, using
     *  `std::vector`s.
     *
     * @param xyz A `std::vector` array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively. If `xyz` it contains a
     *        single point, the class will call a simpler functions that
     *        directly evaluates the point, without a loop.
     * @param sph On entry, a (possibly uninitialized) `std::vector`, which will
     *        be resized to `n_samples * (l_max + 1) * (l_max + 1)`. On exit,
     *        this array will contain the spherical harmonics organized along
     *        two dimensions. The leading dimension is `n_samples` long and it
     *        represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     */
    void compute(const std::vector<T>& xyz, std::vector<T>& sph);

    /** Computes the spherical harmonics and their derivatives with respect to
     *  the Cartesian coordinates of one or more 3D points, using `std::vector`s.
     *
     * @param xyz A `std::vector` array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively. If `xyz` it contains a
     *        single point, the class will call a simpler functions that
     *        directly evaluates the point, without a loop.
     * @param sph On entry, a (possibly uninitialized) `std::vector`, which will
     *        be resized to `n_samples * (l_max + 1) * (l_max + 1)`. On exit,
     *        this array will contain the spherical harmonics organized along
     *        two dimensions. The leading dimension is `n_samples` long and it
     *        represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph `std::vector` for spherical harmonics derivatives.
     *        If not provided, no derivatives will be calculated.
     *        If provided, it is a (possibly uninitialized) `std::vector`, which
     *        will be resized to `n_samples * 3 * (l_max + 1) * (l_max + 1)`. On
     *        exit, this array will contain the spherical harmonics' derivatives
     *        organized along three dimensions. As for the `sph` parameter, the
     *        leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     */
    void compute_with_gradients(const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph);

    void compute_with_hessians(const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph, std::vector<T>& ddsph);

    /** Computes the spherical harmonics for a set of 3D points using bare arrays.
     *
     * @param xyz An array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param xyz_length Total length of the `xyz` array: `n_samples * 3`.
     * @param sph On entry, an array of size `n_samples * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics organized along
     *        two dimensions. The leading dimension is `n_samples` long and it
     *        represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param sph_length Total length of the `sph` array: `n_samples * (l_max + 1) * (l_max + 1)`.
     * @param dsph Optional array for spherical harmonics derivatives.
     *        If not provided, no derivatives will be calculated.
     *        If provided, it is an array of size `n_samples * 3 * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics' derivatives
     *        organized along three dimensions. As for the `sph` parameter, the
     *        leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to the different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     * @param sph_length Optional (only to be provided if `dsph` is).
     *        Total length of the `dsph` array: `n_samples * 3 * (l_max + 1) * (l_max + 1)`.
     */
    void compute_array(const T* xyz, size_t xyz_length, T* sph, size_t sph_length);

    /**
    See `compute_array` above.
    */
    void compute_array_with_gradients(const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length);

    void compute_array_with_hessians(const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length, T* ddsph, size_t ddsph_length);

    /** Computes the spherical harmonics for a single 3D point using bare arrays.
     *
     * @param xyz An array of size 3. It contains the
     *        Cartesian coordinates of the 3D point for which the spherical
     *        harmonics are to be computed. x, y, and z coordinates respectively.
     * @param xyz_length Length of the `xyz` array: 3.
     * @param sph On entry, an array of size `(l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics laid out
     *        in lexicographic order. For example,
     *        if `l_max=2`, it will contain the spherical harmonics in the following
     *        order: `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`.
     * @param sph_length Total length of the `sph` array: `(l_max + 1) * (l_max + 1)`.
     * @param dsph Optional array for spherical harmonics derivatives.
     *        If not provided, no derivatives will be calculated.
     *        If provided, it is an array of size `3 * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics' derivatives
     *        organized along two dimensions. The second
     *        dimension's size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The first dimension
     *        corresponds to the different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     * @param sph_length Optional (only to be provided if `dsph` is).
     *        Total length of the `dsph` array: `3 * (l_max + 1) * (l_max + 1)`.
     */
    void compute_sample(const T* xyz, size_t xyz_length, T* sph, size_t sph_length);
    /**
    See `compute_sample` above.
    */
    void compute_sample_with_gradients(const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length);

    void compute_sample_with_hessians(const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length, T* ddsph, size_t ddsph_length);

    /**
    Returns the number of threads used in the calculation
    */
    int get_omp_num_threads() {
        return this->omp_num_threads;
    }

/* @cond */
private:
    size_t l_max; // maximum l value computed by this class
    size_t size_y; // size of the Ylm rows (l_max+1)**2
    size_t size_q; // size of the prefactor-like arrays (l_max+1)*(l_max+2)/2
    bool normalized; // should we normalize the input vectors?
    int omp_num_threads; // number of openmp thread
    T *prefactors; // storage space for prefactor and buffers
    T *buffers;

    // function pointers are used to set up the right functions to be called
    // these are set in the constructor, so that the public compute functions can 
    // be redirected to the right implementation
    void (*_array_no_derivatives)(const T*, T*, T*, T*, int, int, const T*, T*);
    void (*_array_with_derivatives)(const T*, T*, T*, T*, int, int, const T*, T*);
    void (*_array_with_hessians)(const T*, T*, T*, T*, int, int, const T*, T*);
    
    // these compute a single sample
    void (*_sample_no_derivatives)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
    void (*_sample_with_derivatives)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
    void (*_sample_with_hessians)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
/* @endcond */

};

} //namespace sphericart

#endif
