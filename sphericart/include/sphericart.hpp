/** \file sphericart.hpp
 *  Defines the C++ API for `sphericart`. Two classes are available:
 *  `SphericalHarmonics` and `SolidHarmonics`. The former calculates the
 *  real spherical harmonics :math:`Y^m_l: as defined on Wikipedia,
 *  which are homogeneous polynomials of (x/r, y/r, z/r). The latter
 *  calculates the same polynomials but as a function of the Cartesian coordinates
 *  (x, y, z), or, equivalently, :math:`r^l Y^m_l`.
 */

#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include <cstddef>
#include <tuple>
#include <vector>

#ifdef _SPHERICART_INTERNAL_IMPLEMENTATION
#include "macros.hpp"
#include "templates.hpp"
#endif

namespace sphericart {

/**
 * A spherical harmonics calculator.
 *
 * It handles initialization of the prefactors upon initialization and it
 * stores the buffers that are necessary to compute the spherical harmonics
 * efficiently.
 */
template <typename T> class SphericalHarmonics {
  public:
    /** Initialize the SphericalHarmonics class setting maximum degree and
     * normalization
     *
     *  @param l_max
     *      The maximum degree of the spherical harmonics to be calculated.
     */
    SphericalHarmonics(size_t l_max);

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
     *        single point, the class will call a simpler function that
     *        directly evaluates the point, without a loop.
     * @param sph On entry, a (possibly uninitialized) `std::vector`, which, if
     *        needed, will be resized to `n_samples * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the spherical harmonics organized
     *        along two dimensions. The leading dimension is `n_samples` long
     * and it represents the different samples, while the inner dimension is
     * `(l_max + 1) * (l_max + 1)` long and it contains the spherical harmonics.
     * These are laid out in lexicographic order. For example, if `l_max=2`, it
     * will contain `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
     * (2, 0), (2, 1), (2, 2)`, in this order.
     */
    void compute(const std::vector<T>& xyz, std::vector<T>& sph);

    /** Computes the spherical harmonics and their derivatives with respect to
     *  the Cartesian coordinates of one or more 3D points, using
     * `std::vector`s.
     *
     * @param xyz A `std::vector` array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively. If `xyz` it contains a
     *        single point, the class will call a simpler functions that
     *        directly evaluates the point, without a loop.
     * @param sph On entry, a (possibly uninitialized) `std::vector`, which, if
     *        needed, will be resized to `n_samples * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the spherical harmonics organized
     *        along two dimensions. The leading dimension is `n_samples` long
     * and it represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph `std::vector` for spherical harmonics derivatives.
     *        It is a (possibly uninitialized) `std::vector`, which, if needed,
     *        will be resized to `n_samples * 3 * (l_max + 1) * (l_max + 1)`. On
     *        exit, this array will contain the derivatives of the spherical
     * harmonics organized along three dimensions. As for the `sph` parameter,
     * the leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     */
    void compute_with_gradients(const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph);

    /** Computes the spherical harmonics, their derivatives and second
     * derivatives with respect to the Cartesian coordinates of one or more 3D
     * points, using `std::vector`s.
     *
     * @param xyz A `std::vector` array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively. If `xyz` it contains a
     *        single point, the class will call a simpler functions that
     *        directly evaluates the point, without a loop.
     * @param sph On entry, a (possibly uninitialized) `std::vector`, which, if
     *        needed, will be resized to `n_samples * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the spherical harmonics organized
     *        along two dimensions. The leading dimension is `n_samples` long
     * and it represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph `std::vector` for spherical harmonics derivatives.
     *        It is a (possibly uninitialized) `std::vector`, which, if needed,
     *        will be resized to `n_samples * 3 * (l_max + 1) * (l_max + 1)`. On
     *        exit, this array will contain the derivatives of the spherical
     * harmonics organized along three dimensions. As for the `sph` parameter,
     * the leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     * @param ddsph `std::vector` for spherical harmonics second derivatives.
     *        It is a (possibly uninitialized) `std::vector`, which, if needed,
     *        will be resized to `n_samples * 3 * 3 * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the second derivatives of the
     * spherical harmonics organized along four dimensions. As for the `sph`
     * parameter, the leading dimension represents the different samples, while
     * the inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimensions
     *        correspond to the different spatial second derivatives of the
     * spherical harmonics, i.e., to the dimensions of the Hessian matrix.
     */
    void compute_with_hessians(
        const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph, std::vector<T>& ddsph
    );

    /** Computes the spherical harmonics for a set of 3D points using bare
     * arrays.
     *
     * @param xyz An array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param xyz_length Total length of the `xyz` array: `n_samples * 3`.
     */
    void compute_array(const T* xyz, size_t xyz_length, T* sph, size_t sph_length);

    /** Computes the spherical harmonics and their derivatives for a set of 3D
     * points using bare arrays.
     *
     * @param xyz An array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param xyz_length Total length of the `xyz` array: `n_samples * 3`.
     * @param sph On entry, an array of size `n_samples * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the spherical harmonics organized
     * along two dimensions. The leading dimension is `n_samples` long and it
     *        represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param sph_length Total length of the `sph` array: `n_samples * (l_max +
     * 1)
     * * (l_max + 1)`.
     * @param dsph Array for spherical harmonics derivatives.
     *        It is an array of size `n_samples * 3 * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the derivatives of the spherical
     * harmonics organized along three dimensions. As for the `sph` parameter,
     * the leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to the different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     * @param dsph_length Total length of the `dsph` array: `n_samples * 3 *
     * (l_max + 1) * (l_max + 1)`.
     */
    void compute_array_with_gradients(
        const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length
    );

    /** Computes the spherical harmonics, their derivatives and second
     * derivatives for a set of 3D points using bare arrays.
     *
     * @param xyz An array of size `n_samples x 3`. It contains the
     *        Cartesian coordinates of the 3D points for which the spherical
     *        harmonics are to be computed, organized along two dimensions. The
     *        outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param xyz_length Total length of the `xyz` array: `n_samples * 3`.
     * @param sph On entry, an array of size `n_samples * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the spherical harmonics organized
     * along two dimensions. The leading dimension is `n_samples` long and it
     *        represents the different samples, while the inner dimension size
     *        is `(l_max + 1) * (l_max + 1)` long and it contains the spherical
     *        harmonics. These are laid out in lexicographic order. For example,
     *        if `l_max=2`, it will contain `(l, m) = (0, 0), (1, -1), (1, 0),
     *        (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)`, in this order.
     * @param sph_length Total length of the `sph` array: `n_samples * (l_max +
     * 1)
     * * (l_max + 1)`.
     * @param dsph Array for spherical harmonics derivatives.
     *        It is an array of size `n_samples * 3 * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the derivatives of the spherical
     * harmonics organized along three dimensions. As for the `sph` parameter,
     * the leading dimension represents the different samples, while the
     *        inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimension
     *        corresponds to the different spatial derivatives of the spherical
     *        harmonics: x, y, and z, respectively.
     * @param dsph_length Total length of the `dsph` array: `n_samples * 3 *
     * (l_max + 1) * (l_max + 1)`.
     * @param ddsph Array for spherical harmonics second derivatives.
     *        It is an array of size `n_samples * 3 * 3 * (l_max + 1) * (l_max +
     * 1)`. On exit, this array will contain the second derivatives of the
     * spherical harmonics organized along four dimensions. As for the `sph`
     * parameter, the leading dimension represents the different samples, while
     * the inner-most dimension size is `(l_max + 1) * (l_max + 1)`, and it
     *        represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The intermediate dimensions
     *        correspond to the different spatial second derivatives of the
     * spherical harmonics, i.e., to the dimensions of the Hessian matrix.
     * @param ddsph_length Total length of the `ddsph` array: `n_samples * 9 *
     * (l_max + 1) * (l_max + 1)`.
     */
    void compute_array_with_hessians(
        const T* xyz,
        size_t xyz_length,
        T* sph,
        size_t sph_length,
        T* dsph,
        size_t dsph_length,
        T* ddsph,
        size_t ddsph_length
    );

    /** Computes the spherical harmonics for a single 3D point using bare
     * arrays.
     *
     * @param xyz An array of size 3. It contains the
     *        Cartesian coordinates of the 3D point for which the spherical
     *        harmonics are to be computed. x, y, and z coordinates
     * respectively.
     * @param xyz_length Length of the `xyz` array: 3.
     * @param sph On entry, an array of size `(l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics laid out
     *        in lexicographic order. For example,
     *        if `l_max=2`, it will contain the spherical harmonics in the
     * following order: `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2,
     * -1), (2, 0), (2, 1), (2, 2)`.
     * @param sph_length Total length of the `sph` array: `(l_max + 1) * (l_max
     * + 1)`.
     */
    void compute_sample(const T* xyz, size_t xyz_length, T* sph, size_t sph_length);

    /** Computes the spherical harmonics and their derivatives for a single 3D
     *  point using bare arrays.
     *
     * @param xyz An array of size 3. It contains the
     *        Cartesian coordinates of the 3D point for which the spherical
     *        harmonics are to be computed. x, y, and z coordinates
     * respectively.
     * @param xyz_length Length of the `xyz` array: 3.
     * @param sph On entry, an array of size `(l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics laid out
     *        in lexicographic order. For example,
     *        if `l_max=2`, it will contain the spherical harmonics in the
     * following order: `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2,
     * -1), (2, 0), (2, 1), (2, 2)`.
     * @param sph_length Total length of the `sph` array: `(l_max + 1) * (l_max
     * + 1)`.
     * @param dsph Array for spherical harmonics derivatives.
     *        It is an array of size `3 * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics'
     * derivatives organized along two dimensions. The second dimension's size
     * is
     * `(l_max + 1) * (l_max + 1)`, and it represents the degree and order of
     * the spherical harmonics (again, organized in lexicographic order). The
     * first dimension corresponds to the different spatial derivatives of the
     * spherical harmonics: x, y, and z, respectively.
     * @param dsph_length Total length of the `dsph` array: `3 * (l_max + 1) *
     * (l_max + 1)`.
     */
    void compute_sample_with_gradients(
        const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length
    );

    /** Computes the spherical harmonics, their derivatives and second
     * derivatives for a single 3D point using bare arrays.
     *
     * @param xyz An array of size 3. It contains the
     *        Cartesian coordinates of the 3D point for which the spherical
     *        harmonics are to be computed. x, y, and z coordinates
     * respectively.
     * @param xyz_length Length of the `xyz` array: 3.
     * @param sph On entry, an array of size `(l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics laid out
     *        in lexicographic order. For example,
     *        if `l_max=2`, it will contain the spherical harmonics in the
     * following order: `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2,
     * -1), (2, 0), (2, 1), (2, 2)`.
     * @param sph_length Total length of the `sph` array: `(l_max + 1) * (l_max
     * + 1)`.
     * @param dsph Array for spherical harmonics derivatives.
     *        It is an array of size `3 * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the spherical harmonics'
     * derivatives organized along two dimensions. The second dimension's size
     * is
     * `(l_max + 1) * (l_max + 1)`, and it represents the degree and order of
     * the spherical harmonics (again, organized in lexicographic order). The
     * first dimension corresponds to the different spatial derivatives of the
     * spherical harmonics: x, y, and z, respectively.
     * @param dsph_length Total length of the `dsph` array: `3 * (l_max + 1) *
     * (l_max + 1)`.
     * @param ddsph Array for spherical harmonics second derivatives.
     *        It is an array of size `3 * 3 * (l_max + 1) * (l_max + 1)`.
     *        On exit, this array will contain the second derivatives of the
     * spherical harmonics organized along three dimensions. As for the `sph`
     * parameter, the inner-most dimension size is `(l_max + 1) * (l_max + 1)`,
     * and it represents the degree and order of the spherical harmonics (again,
     *        organized in lexicographic order). The first two dimensions
     *        correspond to the different spatial second derivatives of the
     * spherical harmonics, i.e., to the dimensions of the Hessian matrix.
     * @param ddsph_length Total length of the `ddsph` array: `9 * (l_max + 1) *
     * (l_max + 1)`.
     */
    void compute_sample_with_hessians(
        const T* xyz,
        size_t xyz_length,
        T* sph,
        size_t sph_length,
        T* dsph,
        size_t dsph_length,
        T* ddsph,
        size_t ddsph_length
    );

    /**
     * Returns the maximum degree of the spherical harmonics computed by this
     * calculator.
     */
    size_t get_l_max() { return this->l_max; }

    /**
    Returns the number of threads used in the calculation
    */
    int get_omp_num_threads() { return this->omp_num_threads; }

    /* @cond */
  private:
    template <typename U> friend class SolidHarmonics;

    size_t l_max;        // maximum l value computed by this class
    size_t size_y;       // size of the Ylm rows (l_max+1)**2
    size_t size_q;       // size of the prefactor-like arrays (l_max+1)*(l_max+2)/2
    int omp_num_threads; // number of openmp thread
    T* prefactors;       // storage space for prefactor and buffers
    T* buffers;

    // function pointers are used to set up the right functions to be called
    // these are set in the constructor, so that the public compute functions
    // can be redirected to the right implementation
    void (*_array_no_derivatives)(const T*, T*, T*, T*, size_t, int, const T*, T*);
    void (*_array_with_derivatives)(const T*, T*, T*, T*, size_t, int, const T*, T*);
    void (*_array_with_hessians)(const T*, T*, T*, T*, size_t, int, const T*, T*);

    // these compute a single sample
    void (*_sample_no_derivatives)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
    void (*_sample_with_derivatives)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
    void (*_sample_with_hessians)(const T*, T*, T*, T*, int, int, const T*, const T*, T*, T*, T*);
    /* @endcond */
};

/**
 * A solid harmonics calculator.
 *
 * Its interface is the same as that of the `SphericalHarmonics` class, but it
 * calculates the solid harmonics :math:`r^l Y^m_l` instead of the real spherical
 * harmonics :math:`Y^m_l`, allowing for faster computations.
 */
template <typename T> class SolidHarmonics : public SphericalHarmonics<T> {
  public:
    /** Initialize the SolidHarmonics class setting maximum degree and
     * normalization
     *
     *  @param l_max
     *      The maximum degree of the solid harmonics to be calculated.
     */
    SolidHarmonics(size_t l_max);
};

} // namespace sphericart

#endif
