/** \file sphericart_cuda.hpp
 *  Defines the CUDA C++ API for `sphericart`. Two classes are available:
 *  `SphericalHarmonics` and `SolidHarmonics`. The former calculates the
 *  real spherical harmonics \f$ Y^m_l \f$ as defined on Wikipedia,
 *  which are homogeneous polynomials of (x/r, y/r, z/r). The latter
 *  calculates the same polynomials but as a function of the Cartesian coordinates
 *  (x, y, z), or, equivalently, \f$ r^l\,Y^m_l \f$.
 */

#ifndef SPHERICART_CUDA_HPP
#define SPHERICART_CUDA_HPP

#include <mutex>

// wrap this with cond because breathe can't handle the same namespace in two
// files
/* @cond */
namespace sphericart {
/* @endcond */

/**
 * The `sphericart::cuda` namespace contains the CUDA API for `sphericart`.
 */
namespace cuda {

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

    /** Default constructor
     * Required so sphericart_torch can conditionally instantiate  this class
     * depending on if cuda is available.
     */

    /* @cond */
    ~SphericalHarmonics();
    /* @endcond */

    /** Computes the spherical harmonics for one or more 3D points, using
     *  pre-allocated device-side pointers
     *
     * @param xyz A pre-allocated device-side array of size `n_samples x 3`. It
     * contains the Cartesian coordinates of the 3D points for which the
     * spherical harmonics are to be computed, organized along two dimensions.
     * The outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param n_samples Number of samples contained within `xyz`.
     * @param sph On entry, a preallocated device-side array of size  `n_samples
     * x (l_max + 1) x (l_max + 1)`. On exit, this array will contain the
     * spherical harmonics organized along two dimensions. The leading dimension
     * is `n_samples` long and it represents the different samples, while the
     * inner dimension is
     * `(l_max + 1) x (l_max + 1)` long and it contains the spherical harmonics.
     * These are laid out in lexicographic order. For example, if `l_max=2`, it
     * will contain `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
     * (2, 0), (2, 1), (2, 2)`, in this order.
     * @param cuda_stream Pointer to a cudaStream_t or nullptr. If this is
     * nullptr, the kernel launch will be performed on the default stream.
     */
    void compute(const T* xyz, const size_t n_samples, T* sph, void* cuda_stream = nullptr);

    /** Computes the spherical harmonics and their first derivatives for one or
     *  more 3D points, using pre-allocated device-side pointers
     *
     * @param xyz A pre-allocated device-side array of size `n_samples x 3`. It
     * contains the Cartesian coordinates of the 3D points for which the
     * spherical harmonics are to be computed, organized along two dimensions.
     * The outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param n_samples Number of samples contained within `xyz`.
     * @param sph On entry, a preallocated device-side array of size  `n_samples
     * * (l_max + 1) x (l_max + 1)`. On exit, this array will contain the
     * spherical harmonics organized along two dimensions. The leading dimension
     * is `n_samples` long and it represents the different samples, while the
     * inner dimension is
     * `(l_max + 1) x (l_max + 1)` long and it contains the spherical harmonics.
     * These are laid out in lexicographic order. For example, if `l_max=2`, it
     * will contain `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
     * (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph On entry, nullptr or a preallocated device-side array of size
     * `n_samples x 3 x (l_max + 1) x (l_max + 1)`. If the pointer is not
     * nullptr, then compute_with_gradients must also be true in order for
     * gradients to be computed.
     * @param cuda_stream Pointer to a cudaStream_t or nullptr. If this is
     * nullptr, the kernel launch will be performed on the default stream.
     */
    void compute_with_gradients(
        const T* xyz, const size_t n_samples, T* sph, T* dsph, void* cuda_stream = nullptr
    );

    /** Computes the spherical harmonics and their first and second derivatives
     *  for one or more 3D points, using pre-allocated device-side pointers
     *
     * @param xyz A pre-allocated device-side array of size `n_samples x 3`. It
     * contains the Cartesian coordinates of the 3D points for which the
     * spherical harmonics are to be computed, organized along two dimensions.
     * The outer dimension is `n_samples` long, accounting for different
     *        samples, while the inner dimension has size 3 and it represents
     *        the x, y, and z coordinates respectively.
     * @param n_samples Number of samples contained within `xyz`.
     * @param compute_with_gradients Whether we should compute dsph. If true,
     * the pointer dsph must also be allocated on device.
     * @param compute_with_hessians Whether we should compute ddsph. If true,
     * the pointer ddsph must also be allocated on device.
     * @param sph On entry, a preallocated device-side array of size  `n_samples
     * * (l_max + 1) x (l_max + 1)`. On exit, this array will contain the
     * spherical harmonics organized along two dimensions. The leading dimension
     * is `n_samples` long and it represents the different samples, while the
     * inner dimension is
     * `(l_max + 1) x (l_max + 1)` long and it contains the spherical harmonics.
     * These are laid out in lexicographic order. For example, if `l_max=2`, it
     * will contain `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
     * (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph On entry, nullptr or a preallocated device-side array of size
     * `n_samples x 3 x (l_max + 1) x (l_max + 1)`. If the pointer is not
     * nullptr, then compute_with_gradients must also be true in order for
     * gradients to be computed.
     * @param ddsph On entry, nullptr or a preallocated device-side array of
     * size `n_samples x 3 x 3 x (l_max + 1) x (l_max + 1)`. If the pointer is
     * not nullptr, then compute_with_hessians must also be true in order for
     * gradients to be computed.
     * @param cuda_stream Pointer to a cudaStream_t or nullptr. If this is
     * nullptr, the kernel launch will be performed on the default stream.
     */
    void compute_with_hessians(
        const T* xyz, const size_t n_samples, T* sph, T* dsph, T* ddsph, void* cuda_stream = nullptr
    );

    template <typename U> friend class SolidHarmonics;
    /* @cond */
  private:
    size_t l_max; // maximum l value computed by this class
    size_t nprefactors;
    bool normalized;               // should we normalize the input vectors?
    T* prefactors_cpu = nullptr;   // host prefactors buffer
    T** prefactors_cuda = nullptr; // storage space for prefactors
    int device_count = 0;          // number of visible GPU devices
    int64_t CUDA_GRID_DIM_X_ = 8;
    int64_t CUDA_GRID_DIM_Y_ = 8;
    bool cached_compute_with_gradients = false;
    bool cached_compute_with_hessian = false;
    int64_t _current_shared_mem_allocation = 0;

    void compute_internal(
        const T* xyz,
        const size_t n_samples,
        bool compute_with_gradients,
        bool compute_with_hessian,
        T* sph,
        T* dsph,
        T* ddsph,
        void* cuda_stream
    );
    /* @endcond */
};

/**
 * A solid harmonics calculator.
 *
 * Its interface is the same as that of the `SphericalHarmonics` class, but it
 * calculates the solid harmonics \f$ r^l\,Y^m_l \f$ instead of the real spherical
 * harmonics \f$ Y^m_l \f$, allowing for faster computations.
 */
template <typename T> class SolidHarmonics : public SphericalHarmonics<T> {
  public:
    /** Initialize the SolidHarmonics class setting its maximum degree
     *
     *  @param l_max
     *      The maximum degree of the spherical harmonics to be calculated.
     */
    SolidHarmonics(size_t l_max);
};

} // namespace cuda

/* @cond */
} // namespace sphericart
/* @endcond */

#endif
