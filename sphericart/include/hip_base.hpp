#ifndef SPHERICART_HIP_BASE_HPP
#define SPHERICART_HIP_BASE_HPP

#include <cstdint>

namespace sphericart {

namespace hip {

/**
 * host function which wraps a HIP kernel to compute spherical harmonics and
 * their derivatives.
 *
 * The kernel source is the same CUDA-dialect implementation used by the CUDA
 * backend; it is compiled at runtime with hipRTC (following the mechanism
 * qualified by symmetrix-xl's HIP backend) and launched through
 * hipModuleLaunchKernel.
 *
 * @param xyz
 *        Pointer to a contiguous device-allocated coordinates array of shape
 *        [3N].
 * @param nedges
 *        Number of edges (N).
 * @param prefactors
 *        Prefactors for spherical harmonics recursion.
 * @param nprefactors
 *        Number of elements in the prefactors array.
 * @param l_max
 *        The maximum degree of the spherical harmonics to be calculated.
 * @param normalize
 *        If `false` (default) computes the scaled spherical harmonics, which
 *        are homogeneous polynomials in the Cartesian coordinates of the input
 *        points. If `true`, computes the normalized spherical harmonics that
 *        are evaluated on the unit sphere. In practice, this simply computes
 *        the scaled harmonics at the normalized coordinates \f$(x/r, y/r, z/r)\f$, and
 *        adapts the derivatives accordingly.
 * @param GRID_DIM_X
 *        The size of the threadblock in the x dimension. Used to parallelize
 *        over the sample dimension
 * @param GRID_DIM_Y
 *        The size of the threadblock in the y dimension. Used only to improve
 *          memory throughput on reads and writes.
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
 * @param hip_stream
 *        Pointer to a hipStream_t or nullptr. If this is nullptr, the kernel
 *        launch will be performed on the default stream.
 */
template <typename scalar_t>
void spherical_harmonics_hip_base(
    const scalar_t* xyz,
    const int nedges,
    const scalar_t* prefactors,
    const int nprefactors,
    const int l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    void* hip_stream = nullptr,
    bool synchronize = true
);

template <typename scalar_t>
void spherical_harmonics_backward_hip_base(
    const scalar_t* dsph,
    const scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* hip_stream = nullptr
);

} // namespace hip
} // namespace sphericart

#endif
