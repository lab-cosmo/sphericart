/** \file sphericart_cuda.hpp
 *  Defines the CUDA API for `sphericart`.
 */
#ifndef SPHERICART_CUDA_HPP
#define SPHERICART_CUDA_HPP

#include <mutex>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

namespace sphericart {

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
     *  @param normalized
     *      If `false` (default) computes the scaled spherical harmonics, which
     * are homogeneous polynomials in the Cartesian coordinates of the input
     * points. If `true`, computes the normalized spherical harmonics that are
     * evaluated on the unit sphere. In practice, this simply computes the
     * scaled harmonics at the normalized coordinates \f$(x/r, y/r, z/r)\f$, and
     * adapts the derivatives accordingly.
     */
    SphericalHarmonics(size_t l_max, bool normalized = false);

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
     * @param nsamples Number of samples contained within `xyz`.
     * @param compute_with_gradients Whether we should compute dsph. If true,
     * the pointer dsph must also be allocated on device.
     * @param compute_with_hessians Whether we should compute ddsph. If true,
     * the pointer ddsph must also be allocated on device.
     * @param sph On entry, a preallocated device-side array of size  `n_samples
     * * (l_max + 1) * (l_max + 1)`. On exit, this array will contain the
     * spherical harmonics organized along two dimensions. The leading dimension
     * is `n_samples` long and it represents the different samples, while the
     * inner dimension is
     * `(l_max + 1) * (l_max + 1)` long and it contains the spherical harmonics.
     * These are laid out in lexicographic order. For example, if `l_max=2`, it
     * will contain `(l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
     * (2, 0), (2, 1), (2, 2)`, in this order.
     * @param dsph On entry, nullptr or a preallocated device-side array of size
     * `n_samples * 3 * (l_max + 1) * (l_max + 1)`. If the pointer is not
     * nullptr, then compute_with_gradients must also be true in order for
     * gradients to be computed.
     * @param ddsph On entry, nullptr or a preallocated device-side array of
     * size `n_samples * 3 * 3 * (l_max + 1) * (l_max + 1)`. If the pointer is
     * not nullptr, then compute_with_hessians must also be true in order for
     * gradients to be computed.
     * @param cuda_stream Pointer to a cudaStream_t or nullptr. If this is
     * nullptr, the kernel launch will be performed on the default stream.
     */

    void compute(const T *xyz, size_t nsamples, bool compute_with_gradients,
                 bool compute_with_hessian, T *sph, T *dsph = nullptr,
                 T *ddsph = nullptr, void *cuda_stream = nullptr);

  private:
    size_t l_max; // maximum l value computed by this class
    size_t nprefactors;
    bool normalized;    // should we normalize the input vectors?
    T *prefactors_cpu;  // host prefactors buffer
    T *prefactors_cuda; // storage space for prefactors

    int64_t CUDA_GRID_DIM_X_ = 8;
    int64_t CUDA_GRID_DIM_Y_ = 8;

    int _current_shared_mem_allocation = 0;
};

} // namespace cuda
} // namespace sphericart
#endif
