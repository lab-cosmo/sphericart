#ifndef SPHERICART_CUDA_BASE_HPP
#define SPHERICART_CUDA_BASE_HPP

namespace sphericart {

namespace cuda {

/**
 * host function which wraps a CUDA kernel to compute spherical harmonics and
 * their derivatives.
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
 * @param GRID_DIM_X
 *        The size of the threadblock in the x dimension. Used to parallelize
 *        over the sample dimension
 * @param GRID_DIM_Y
 *        The size of the threadblock in the y dimension. Used only to improve
 *          memory throughput on reads and writes.
 * @param xyz_requires_grad
 *        Boolean representing whether or not the input XYZ requires grad -
 *        required for torch.
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
 * @param cuda_stream
 *        Pointer to a cudaStream_t or nullptr. If this is nullptr, the kernel
 * launch will be performed on the default stream.
 */
template <typename scalar_t>
void spherical_harmonics_cuda_base(
    const scalar_t* xyz,
    const int nedges,
    scalar_t* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    void* cuda_stream = nullptr
);

template <typename scalar_t>
void spherical_harmonics_backward_cuda_base(
    scalar_t* dsph,
    scalar_t* sph_grad,
    const int nedges,
    const int ntotal,
    scalar_t* xyz_grad,
    void* cuda_stream = nullptr
);

/**
 * Host function to ensure the current kernel launch parameters have sufficient
 * shared memory given by the default space provided by the card. Returns the
 * amount of bytes thats in use for shared memory, after attempting to adjust
 * it's size if necessary. Returns -1 if too much shared memory is requested.
 *
 * @param element_size
 *        the number of bytes of the scalar type used in the input/output arrays
 *        (4 or 8).
 * @param l_max
 *        The maximum degree of the spherical harmonics to be calculated.
 * @param GRID_DIM_X
 *        The size of the threadblock in the x dimension.
 * @param GRID_DIM_Y
 *        The size of the threadblock in the y dimension.
 * @param requires_grad
 *        Boolean representing if we need first-order derivatives.
 * @param requires_hessian
 *        Boolean representing if we need second-order derivatives.
 * @param current_shared_mem_alloc
 *        the current size of the shared memory allocation.
 */
int adjust_shared_memory(
    size_t element_size,
    int64_t l_max,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool requires_grad,
    bool requires_hessian,
    int64_t current_shared_mem_alloc
);

} // namespace cuda
} // namespace sphericart

#endif
