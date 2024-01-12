/** \file sphericart_cuda.hpp
 *  Defines the CUDA API for `sphericart`.
 */
#ifndef SPHERICART_CUDA_HPP
#define SPHERICART_CUDA_HPP

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
     * @param xyz todo docs
     * @param sph todo docs
     */
    void compute(const T *xyz, size_t nsamples, bool compute_with_gradients,
                 bool compute_with_hessian, size_t GRID_DIM_X,
                 size_t GRID_DIM_Y, T *sph, T *dsph = nullptr,
                 T *ddsph = nullptr);

  private:
    size_t l_max;       // maximum l value computed by this class
    bool normalized;    // should we normalize the input vectors?
    T *prefactors_cpu;  // host prefactors buffer
    T *prefactors_cuda; // storage space for prefactors
};

} // namespace cuda
} // namespace sphericart
#endif
