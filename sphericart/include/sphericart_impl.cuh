#ifndef SPHERICART_IMPL_CUH
#define SPHERICART_IMPL_CUH

/*
    Clears the shared memory buffers for the spherical harmonics and gradients
   if required.
*/
template <typename scalar_t>
__device__ inline void clear_buffers(
    int nelements,
    scalar_t* sph,
    scalar_t* dsph_x,
    scalar_t* dsph_y,
    scalar_t* dsph_z,

    scalar_t* dsph_dxdx,
    scalar_t* dsph_dxdy,
    scalar_t* dsph_dxdz,

    scalar_t* dsph_dydx,
    scalar_t* dsph_dydy,
    scalar_t* dsph_dydz,

    scalar_t* dsph_dzdx,
    scalar_t* dsph_dzdy,
    scalar_t* dsph_dzdz,
    bool requires_grad,
    bool requires_hessian
);

/*
    Writes out the shared memory buffers to global memory, as well as applying
   normalisation if necessary.
*/
template <typename scalar_t>
__device__ inline void write_buffers(
    int edge_idx,
    int nedges,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t ir,
    int n_elements,
    int offset,
    scalar_t* buffer_sph,

    scalar_t* buffer_dsph_x,
    scalar_t* buffer_dsph_y,
    scalar_t* buffer_dsph_z,

    scalar_t* buffer_dsph_dxdx,
    scalar_t* buffer_dsph_dxdy,
    scalar_t* buffer_dsph_dxdz,

    scalar_t* buffer_dsph_dydx,
    scalar_t* buffer_dsph_dydy,
    scalar_t* buffer_dsph_dydz,

    scalar_t* buffer_dsph_dzdx,
    scalar_t* buffer_dsph_dzdy,
    scalar_t* buffer_dsph_dzdz,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    int n_total,
    bool requires_grad,
    bool requires_hessian,
    bool normalize
);

/*
    CUDA kernel for computing Cartesian spherical harmonics and their
   derivatives.
*/
template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    scalar_t* xyz,
    int nedges,
    scalar_t* prefactors,
    int nprefactors,
    int lmax,
    int ntotal,
    bool requires_grad,
    bool requires_hessian,
    bool normalize,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph
);

/*
    CUDA kernel to computes the backwards pass for autograd.
*/
template <typename scalar_t>
__global__ void backward_kernel(
    scalar_t* dsph, scalar_t* sph_grad, int nedges, int n_total, scalar_t* xyz_grad
);

#endif