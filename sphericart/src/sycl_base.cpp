#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include <cmath>
#include "sycl_base.hpp"
#include "sphericart_impl_sycl.hpp"
#include <algorithm>

#define HARDCODED_LMAX 1

/*
    Computes the total amount of shared memory space required by
   spherical_harmonics_kernel.

    For lmax <= HARCODED_LMAX, we need to store all (HARDCODED_LMAX + 1)**2
   scalars in shared memory. For lmax > HARDCODED_LMAX, we only need to store
   each spherical harmonics vector per sample in shared memory.
*/
template <typename scalar_t>
void sphericart::sycl::spherical_harmonics_sycl_base(
    const scalar_t* xyz,
    const int nedges,
    const scalar_t* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph
) {
    spherical_harmonics_kernel(
        xyz, nedges, prefactors, l_max, gradients, hessian, normalize, sph, dsph, ddsph
    );
}

template void sphericart::sycl::spherical_harmonics_sycl_base<float>(
    const float* xyz,
    const int nedges,
    const float* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    float* sph,
    float* dsph,
    float* ddsph
);

template void sphericart::sycl::spherical_harmonics_sycl_base<double>(
    const double* xyz,
    const int nedges,
    const double* prefactors,
    const int nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    double* sph,
    double* dsph,
    double* ddsph
);

template <typename scalar_t>
void sphericart::sycl::spherical_harmonics_backward_sycl_base(
    const scalar_t* dsph, const scalar_t* sph_grad, const int nedges, const int ntotal, scalar_t* xyz_grad
) {
    backward_kernel(dsph, sph_grad, nedges, ntotal, xyz_grad);
}

template void sphericart::sycl::spherical_harmonics_backward_sycl_base<float>(
    const float* dsph, const float* sph_grad, const int nedges, const int ntotal, float* xyz_grad
);

template void sphericart::sycl::spherical_harmonics_backward_sycl_base<double>(
    const double* dsph, const double* sph_grad, const int nedges, const int ntotal, double* xyz_grad
);
