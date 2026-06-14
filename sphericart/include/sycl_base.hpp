#ifndef SPHERICART_SYCL_BASE_HPP
#define SPHERICART_SYCL_BASE_HPP

#include "sycl_device.hpp"

namespace sphericart {
namespace sycl {

/**
 * Host function which launches the SYCL kernel to compute spherical harmonics
 * and their derivatives.
 *
 * @param xyz
 *        Pointer to a contiguous device-allocated coordinates array of shape
 *        [3N].
 * @param nedges
 *        Number of edges (N).
 * @param prefactors
 *        Prefactors for spherical harmonics recursion.
 * @param l_max
 *        The maximum degree of the spherical harmonics to be calculated.
 * @param normalize
 *        If `false` (default) computes the scaled spherical harmonics, which
 *        are homogeneous polynomials in the Cartesian coordinates of the input
 *        points. If `true`, computes the normalized spherical harmonics that
 *        are evaluated on the unit sphere. In practice, this simply computes
 * the scaled harmonics at the normalized coordinates \f$(x/r, y/r, z/r)\f$, and
 * adapts the derivatives accordingly.
 * @param gradients
 *        Perform the computation of the first-order derivatives.
 * @param hessian
 *        Perform the computation of the second-order derivatives.
 * @param sph
 *        Pointer to a contiguous device-allocated spherical harmonics array of
 *        shape [N * (L + 1) ** 2].
 * @param dsph
 *        Pointer to a contiguous device-allocated spherical harmonics
 *        first-derivatives array of shape [N * 3 * (L + 1) ** 2].
 * @param ddsph
 *        Pointer to a contiguous device-allocated spherical harmonics
 *        second-derivatives array of shape [N * 3 * (L + 1) ** 2].
 * launch will be performed on the default stream.
 */
template <typename scalar_t>
void spherical_harmonics_sycl_base(
    const scalar_t* xyz,
    const int nedges,
    const scalar_t* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph
);

template <typename scalar_t>
void spherical_harmonics_backward_sycl_base(
    const scalar_t* dsph, const scalar_t* sph_grad, const int nedges, const int ntotal, scalar_t* xyz_grad
);

} // namespace sycl
} // namespace sphericart

#endif
