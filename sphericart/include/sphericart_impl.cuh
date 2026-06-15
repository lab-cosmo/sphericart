#ifndef SPHERICART_IMPL_CUH
#define SPHERICART_IMPL_CUH

#define CUDA_DEVICE_PREFIX __device__

#if !defined(__CUDACC__) && !defined(__global__)
#define __global__
#endif

/*
    CUDA kernel for computing Cartesian spherical harmonics and their
   derivatives.
*/
template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    const scalar_t* xyz,
    int nedges,
    const scalar_t* prefactors,
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
    const scalar_t* dsph, const scalar_t* sph_grad, int nedges, int n_total, scalar_t* xyz_grad
);

#endif
