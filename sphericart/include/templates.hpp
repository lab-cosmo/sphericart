#ifndef SPHERICART_TEMPLATES_HPP
#define SPHERICART_TEMPLATES_HPP

#ifndef CUDA_DEVICE_PREFIX
#define CUDA_DEVICE_PREFIX
#endif

/*
    Template implementation of Cartesian Ylm calculators.

    The template functions use compile-time `if constexpr()` constructs to
    implement calculators for spherical harmonics that can handle different
    type of calls, e.g. with or without derivative calculations, and with
    different numbers of terms computed with hard-coded expressions.
*/

// Defines the largest l_max for which we have templated implementations,
// which can benefit from loop unrolling
#define SPHERICART_LMAX_TEMPLATED 10

#include <cmath>
#include <iostream>
#include <vector>

#ifdef _OPENMP

#include <omp.h>

#else
// define dummy versions of the functions we need

static inline int omp_get_max_threads() { return 1; }

static inline int omp_get_thread_num() { return 0; }

#endif

// a SPH_IDX that does nothing
#define DUMMY_SPH_IDX

/**
 * This function calculates the prefactors needed for the computation of the
 * spherical harmonics.
 *
 * @param l_max The maximum degree of spherical harmonics for which the
 *        prefactors will be calculated.
 * @param factors On entry, a (possibly uninitialized) array of size
 *        `(l_max+1) * (l_max+2)`. On exit, it will contain the prefactors for
 *        the calculation of the spherical harmonics up to degree `l_max`, in
 *        the order `(l, m) = (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), ...`.
 *        The array contains two blocks of size `(l_max+1) * (l_max+2) / 2`:
 *        the first holds the numerical prefactors that enter the full
 * \f$Y_l^m\f$, the second containing constansts that are needed to evaluate
 * the \f$Q_l^m\f$.
 */
template <typename T> void compute_sph_prefactors(int l_max, T *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
        Use an iterative formula to avoid computing a ratio
        of factorials, and incorporates the 1/sqrt(2) that
        is associated with the Yl0's
        Also computes a set of coefficients that are needed
        in the iterative calculation of the Qlm, and just
        stashes them at the end of factors, which should therefore
        be (l_max+1)*(l_max+2) in size
    */

    auto k = 0; // quick access index
    for (int l = 0; l <= l_max; ++l) {
        T factor = (2 * l + 1) / (2 * static_cast<T>(M_PI));
        // incorporates  the 1/sqrt(2) that goes with the m=0 SPH
        factors[k] = std::sqrt(factor) * static_cast<T>(M_SQRT1_2);
        for (int m = 1; m <= l; ++m) {
            factor *= static_cast<T>(1.0) / (l * (l + 1) + m * (1 - m));
            if (m % 2 == 0) {
                factors[k + m] = std::sqrt(factor);
            } else {
                factors[k + m] = -std::sqrt(factor);
            }
        }
        k += l + 1;
    }

    // that are needed in the recursive calculation of Qlm.
    // Xll is just Qll, Xlm is the factor that enters the alternative m
    // recursion
    factors[k] = 1;
    k += 1;
    for (int l = 1; l < l_max + 1; l++) {
        factors[k + l] = -(2 * l - 1) * factors[k - 1];
        for (int m = l - 1; m >= 0; --m) {
            factors[k + m] = static_cast<T>(-1.0) / ((l + m + 1) * (l - m));
        }
        k += l + 1;
    }
}

template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          bool NORMALIZED, int HARDCODED_LMAX>
inline void hardcoded_sph_sample(const T *xyz_i, T *sph_i,
                                 [[maybe_unused]] T *dsph_i,
                                 [[maybe_unused]] T *ddsph_i,
                                 [[maybe_unused]] int l_max_dummy =
                                     0, // dummy variables to have a uniform
                                        // interface with generic_sph_ functions
                                 [[maybe_unused]] int size_y = 1,
                                 [[maybe_unused]] const T *py_dummy = nullptr,
                                 [[maybe_unused]] const T *qy_dummy = nullptr,
                                 [[maybe_unused]] T *c_dummy = nullptr,
                                 [[maybe_unused]] T *s_dummy = nullptr,
                                 [[maybe_unused]] T *z_dummy = nullptr) {
    /*
        Wrapper for the hardcoded macros that also allows to apply
       normalization. Computes a single sample, and uses a template to avoid
       branching.

        Template parameters:
        typename T: float type (e.g. single/double precision)
        bool DO_DERIVATIVES: should se evaluate the derivatives?
        bool DO_SECOND_DERIVATIVES: should se evaluate the second derivatives?
        bool NORMALIZED: should we normalize the input positions?
        int HARDCODED_LMAX: which lmax value will be computed

        NB: this is meant to be computed for a maximum LMAX value defined at
       compile time. the l_max_dummy parameter (that correspond to l_max in the
       generic implementation) is ignored

        Actual parameters:
        const T *xyz_i: a T array containing the x,y,z coordinates of a single
       sample (x,y,z) T *sph_i: pointer to the storage location for the Ylm
       (stored as l,m= (0,0),(1,-1),(1,0),(1,1),...
        [[maybe_unused]] T *dsph_i : pointer to the storage location for the
       dYlm/dx,dy,dz. stored as for sph_i, with three consecutive blocks
       associated to d/dx,d/dy,d/dz
        [[maybe_unused]] T *ddsph_i : pointer to the storage location for the
       second derivatives. stored as for sph_i, with nine consecutive blocks
       associated to the nine second derivative combinations
        [[maybe_unused]] int size_y: size of storage for the y,
       (HARDCODED_LMAX+1)**2

        ALL XXX_dummy variables are defined to match the interface of
       generic_sph_sample and are ignored
    */

    static_assert(
        !(DO_SECOND_DERIVATIVES && !DO_DERIVATIVES),
        "Cannot calculate second derivatives without first derivatives");
    static_assert(
        !(DO_SECOND_DERIVATIVES && HARDCODED_LMAX > 1),
        "Hardcoded second derivatives are only implemented up to l=1.");

    auto x = xyz_i[0];
    auto y = xyz_i[1];
    auto z = xyz_i[2];
    [[maybe_unused]] auto x2 = x * x;
    [[maybe_unused]] auto y2 = y * y;
    [[maybe_unused]] auto z2 = z * z;
    [[maybe_unused]] T ir = 0.0; // 1/r, it is only computed and used if we
                                 // normalize the input vector

    if constexpr (NORMALIZED) {
        ir = 1 / std::sqrt(x2 + y2 + z2);
        x *= ir;
        y *= ir;
        z *= ir;
        x2 = x * x;
        y2 = y * y;
        z2 = z * z;
    }

    // nb: asserting that HARDCODED_LMAX is not too large is done statically
    // inside the macro
    HARDCODED_SPH_MACRO(HARDCODED_LMAX, x, y, z, x2, y2, z2, sph_i,
                        DUMMY_SPH_IDX);

    if constexpr (DO_DERIVATIVES) {
        // computes the derivatives
        T *dx_sph_i = dsph_i;
        T *dy_sph_i = dx_sph_i + size_y;
        T *dz_sph_i = dy_sph_i + size_y;
        HARDCODED_SPH_DERIVATIVE_MACRO(HARDCODED_LMAX, x, y, z, x2, y2, z2,
                                       sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
                                       DUMMY_SPH_IDX);

        if constexpr (DO_SECOND_DERIVATIVES) {
            // set each second derivative pointer to the appropriate place
            T *dxdx_sph_i = ddsph_i;
            T *dxdy_sph_i = dxdx_sph_i + size_y;
            T *dxdz_sph_i = dxdy_sph_i + size_y;
            T *dydx_sph_i = dxdz_sph_i + size_y;
            T *dydy_sph_i = dydx_sph_i + size_y;
            T *dydz_sph_i = dydy_sph_i + size_y;
            T *dzdx_sph_i = dydz_sph_i + size_y;
            T *dzdy_sph_i = dzdx_sph_i + size_y;
            T *dzdz_sph_i = dzdy_sph_i + size_y;
            HARDCODED_SPH_SECOND_DERIVATIVE_MACRO(
                HARDCODED_LMAX, sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                dzdz_sph_i, DUMMY_SPH_IDX);

            if constexpr (NORMALIZED) {
                for (int k = 0; k < size_y; ++k) {
                    // We loop again over k (and recalculate tmp for the second
                    // derivatives) to avoid crazy nesting of these sections.
                    // The main issue is that if(constexpr) restricts the scope
                    // of the double derivative pointers.

                    // correct second derivatives for normalization. We do it
                    // before the first derivatives because we need the
                    // unchanged first derivatives
                    auto irsq = ir * ir;
                    auto tmp =
                        (dx_sph_i[k] * x + dy_sph_i[k] * y + dz_sph_i[k] * z);
                    auto tmpx = x * dxdx_sph_i[k] + y * dydx_sph_i[k] +
                                z * dzdx_sph_i[k];
                    auto tmpy = x * dxdy_sph_i[k] + y * dydy_sph_i[k] +
                                z * dydz_sph_i[k];
                    auto tmpz = x * dxdz_sph_i[k] + y * dydz_sph_i[k] +
                                z * dzdz_sph_i[k];
                    auto tmp2 =
                        x * x * dxdx_sph_i[k] + y * y * dydy_sph_i[k] +
                        z * z * dzdz_sph_i[k] + 2 * x * y * dxdy_sph_i[k] +
                        2 * x * z * dxdz_sph_i[k] + 2 * y * z * dydz_sph_i[k];
                    dxdx_sph_i[k] =
                        (-2 * x * tmpx + dxdx_sph_i[k] + 3 * x * x * tmp - tmp -
                         2 * x * dx_sph_i[k] + x * x * tmp2) *
                        irsq;
                    dydy_sph_i[k] =
                        (-2 * y * tmpy + dydy_sph_i[k] + 3 * y * y * tmp - tmp -
                         2 * y * dy_sph_i[k] + y * y * tmp2) *
                        irsq;
                    dzdz_sph_i[k] =
                        (-2 * z * tmpz + dzdz_sph_i[k] + 3 * z * z * tmp - tmp -
                         2 * z * dz_sph_i[k] + z * z * tmp2) *
                        irsq;
                    dxdy_sph_i[k] = dydx_sph_i[k] =
                        (-x * tmpy - y * tmpx + dxdy_sph_i[k] +
                         3 * x * y * tmp - x * dy_sph_i[k] - y * dx_sph_i[k] +
                         x * y * tmp2) *
                        irsq;
                    dxdz_sph_i[k] = dzdx_sph_i[k] =
                        (-x * tmpz - z * tmpx + dxdz_sph_i[k] +
                         3 * x * z * tmp - x * dz_sph_i[k] - z * dx_sph_i[k] +
                         x * z * tmp2) *
                        irsq;
                    dzdy_sph_i[k] = dydz_sph_i[k] =
                        (-z * tmpy - y * tmpz + dzdy_sph_i[k] +
                         3 * y * z * tmp - z * dy_sph_i[k] - y * dz_sph_i[k] +
                         y * z * tmp2) *
                        irsq;
                }
            }
        }

        if constexpr (NORMALIZED) {
            // corrects derivatives for normalization
            for (int k = 0; k < size_y; ++k) {
                auto tmp =
                    (dx_sph_i[k] * x + dy_sph_i[k] * y + dz_sph_i[k] * z);
                dx_sph_i[k] = (dx_sph_i[k] - x * tmp) * ir;
                dy_sph_i[k] = (dy_sph_i[k] - y * tmp) * ir;
                dz_sph_i[k] = (dz_sph_i[k] - z * tmp) * ir;
            }
        }
    }
}

template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          bool NORMALIZED, int HARDCODED_LMAX>
void hardcoded_sph(
    const T *xyz, T *sph, [[maybe_unused]] T *dsph, [[maybe_unused]] T *ddsph,
    size_t n_samples,
    [[maybe_unused]] int l_max_dummy =
        0, // dummy variables to have a uniform interface with generic_sph
    [[maybe_unused]] const T *prefactors_dummy = nullptr,
    [[maybe_unused]] T *buffers_dummy = nullptr) {
    /*
        Cartesian Ylm calculator using the hardcoded expressions.
        Templated version, just calls hardcoded_sph_sample within a loop.
        XXX_dummy variables are ignored

        Template parameters: see hardcoded_sph_sample

        Actual parameters:
        const T *xyz: a T array containing th n_samplex*3 x,y,z coordinates of
       multiple 3D points T *sph: pointer to the storage location for the Ylm
       (stored as l,m= (0,0),(1,-1),(1,0),(1,1),...
        [[maybe_unused]] T *dsph : pointer to the storage location for the
       dYlm/dx,dy,dz. stored as for sph_i, with three consecutive blocks
       associated to d/dx,d/dy,d/dz
        [[maybe_unused]] T *ddsph : pointer to the storage location for the
       second derivatives. stored as for sph_i, with nine consecutive blocks
       associated to the nine possible second derivative combinations size_t
       n_samples: number of samples that have to be computed

    */
    static_assert(!(DO_SECOND_DERIVATIVES && HARDCODED_LMAX > 1),
                  "Hardcoded second derivatives are not implemented for l>1.");

    constexpr auto size_y = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);

#pragma omp parallel
    {
        const T *xyz_i = nullptr;
        T *sph_i = nullptr;
        T *dsph_i = nullptr;
        T *ddsph_i = nullptr;

#pragma omp for
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            // gets pointers to the current sample input and output arrays
            xyz_i = xyz + i_sample * 3;
            sph_i = sph + i_sample * size_y;
            if constexpr (DO_DERIVATIVES) {
                dsph_i = dsph + i_sample * size_y * 3;
            }
            if constexpr (DO_SECOND_DERIVATIVES) {
                ddsph_i = ddsph + i_sample * size_y * 9;
            }
            hardcoded_sph_sample<T, DO_DERIVATIVES, DO_SECOND_DERIVATIVES,
                                 NORMALIZED, HARDCODED_LMAX>(
                xyz_i, sph_i, dsph_i, ddsph_i, HARDCODED_LMAX, size_y);
        }
    }
}

int inline dummy_idx(int i) { return i; }

/** Computes the sph and their derivatives for a given Cartesian point and a
 * given l. The template implementation supports different floating point types
 * T, determines whether to compute derivatives (DO_DERIVATIVES), assumes that
 * l is greater than HARDCODED_LMAX. GET_INDEX is a function that might allow
 * to map differently the indices in the spherical harmonics (used in the CUDA
 * implementation).
 */
template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          int HARDCODED_LMAX, int (*GET_INDEX)(int) = dummy_idx>
CUDA_DEVICE_PREFIX static inline void generic_sph_l_channel(
    int l,
    [[maybe_unused]] T x, // these might be unused for low LMAX. not worth a
                          // full separate implementation
    [[maybe_unused]] T y, [[maybe_unused]] T z, [[maybe_unused]] T rxy,
    const T *pk, const T *qlmk, T *c, T *s, T *twomz, T *sph_i,
    [[maybe_unused]] T *dx_sph_i, [[maybe_unused]] T *dy_sph_i,
    [[maybe_unused]] T *dz_sph_i, [[maybe_unused]] T *dxdx_sph_i,
    [[maybe_unused]] T *dxdy_sph_i, [[maybe_unused]] T *dxdz_sph_i,
    [[maybe_unused]] T *dydx_sph_i, [[maybe_unused]] T *dydy_sph_i,
    [[maybe_unused]] T *dydz_sph_i, [[maybe_unused]] T *dzdx_sph_i,
    [[maybe_unused]] T *dzdy_sph_i, [[maybe_unused]] T *dzdz_sph_i) {
    /*
    This is the main low-level code to compute sph and dsph for an arbitrary l.
    The code is a bit hard to follow because of (too?) many micro-optimizations.
    Sine and cosine terms are precomputed. Here the Qlm modifield Legendre
    polynomials are evaluated, and combined with the other terms and the
    prefactors. The core iteration is an iteration down to lower values of m,

    Qlm = A z Ql(m+1) + B rxy^2 Ql(m+2)

    1. the special cases with l=+-m and +-1 are done separately, also because
    they initialize the recursive expression
    2. we assume that some lower l are done with hard-coding, and HARDCODED_LMAX
    is passed as a template parameter. this is used to split the loops over m in
    a part with fixed size, known at compile time, and one with variable length.
    3. we do the recursion using stack variables and never store the full Qlm
    array
    4. we compute separately Qlm and Q(l-1) - the latter needed for derivatives
    rather than reuse the calculation from another l channel. It appears that
    the simplification in memory access makes this beneficial, with the added
    advantage that each l channel can be computed independently

    Template parameters:
    typename T: float type (e.g. single/double precision)
    bool DO_DERIVATIVES: should we evaluate the derivatives?
    bool DO_SECOND_DERIVATIVES: should we evaluate the second derivatives?
    bool NORMALIZED: should we normalize the input positions?
    int HARDCODED_LMAX: maximum value of l for which hardcoding was used

    Actual parameters:
    l: which l we are computing
    x,y,z: the Cartesian coordinates of the point
    rxy: sqrt(x^2+y^2), precomputed because it's used for all l
    pk, qlmk: prefactors used in the calculation of Ylm and Qlm, respectively
    c,s: the c_k and s_k cos-like and sin-like terms combined with the Qlm to
    compute Ylm twomz: 2*m*z, these are also computed once and reused for all l
    sph_i, d[x,y,z]sph_i: storage locations of the output arrays for Ylm and
    dYlm/d[x,y,z] d[x,y,z]d[x,y,z]sph_i: storage locations of the output arrays
    for the second derivatives

    */
    static_assert(
        !(DO_SECOND_DERIVATIVES && !DO_DERIVATIVES),
        "Cannot calculate second derivatives without first derivatives");

    // working space for the recursive evaluation of Qlm and Q(l-1)m.
    // qlm_[0,1,2] correspond to the current Qlm, Ql(m+1) and Ql(m+2), and the
    // ql1m_[0,1,2] hold the same but for l-1
    // ql2m_[0,1,2] hold the same but for l-2
    [[maybe_unused]] T qlm_2, qlm_1, qlm_0;
    [[maybe_unused]] T ql1m_2, ql1m_1, ql1m_0;
    [[maybe_unused]] T ql2m_2, ql2m_1, ql2m_0;

    [[maybe_unused]] T x2, y2,
        xy; // for second derivatives. we could get them from parent but not
            // worth the additional complexity
    if constexpr (DO_SECOND_DERIVATIVES) {
        x2 = x * x;
        y2 = y * y;
        xy = x * y;
    }

    // m=+-l
    qlm_2 = qlmk[l]; // fetches the pre-computed Qll
    auto pq = qlm_2 * pk[l];
    sph_i[GET_INDEX(-l)] = pq * s[GET_INDEX(l)];
    sph_i[GET_INDEX(+l)] = pq * c[GET_INDEX(l)];

    if constexpr (DO_DERIVATIVES) {
        pq *= l;
        dx_sph_i[GET_INDEX(-l)] = pq * s[GET_INDEX(l - 1)];
        dy_sph_i[GET_INDEX(-l)] = dx_sph_i[GET_INDEX(l)] =
            pq * c[GET_INDEX(l - 1)];
        dy_sph_i[GET_INDEX(l)] = -dx_sph_i[GET_INDEX(-l)];
        dz_sph_i[GET_INDEX(-l)] = 0;
        dz_sph_i[GET_INDEX(l)] = 0;
        ql1m_2 = 0;

        if constexpr (DO_SECOND_DERIVATIVES) {
            pq *= (l - 1);
            dxdx_sph_i[GET_INDEX(l)] = pq * c[GET_INDEX(l - 2)];
            dxdx_sph_i[GET_INDEX(-l)] = pq * s[GET_INDEX(l - 2)];
            dxdy_sph_i[GET_INDEX(l)] = dydx_sph_i[GET_INDEX(l)] =
                dydy_sph_i[GET_INDEX(-l)] = -dxdx_sph_i[GET_INDEX(-l)];
            dxdy_sph_i[GET_INDEX(-l)] = dydx_sph_i[GET_INDEX(-l)] =
                dxdx_sph_i[GET_INDEX(l)];
            dxdz_sph_i[GET_INDEX(l)] = dzdx_sph_i[GET_INDEX(l)] = 0.0;
            dxdz_sph_i[GET_INDEX(-l)] = dzdx_sph_i[GET_INDEX(-l)] = 0.0;
            dydy_sph_i[GET_INDEX(l)] = -dxdx_sph_i[GET_INDEX(l)];
            dydz_sph_i[GET_INDEX(l)] = dzdy_sph_i[GET_INDEX(l)] = 0.0;
            dydz_sph_i[GET_INDEX(-l)] = dzdy_sph_i[GET_INDEX(-l)] = 0.0;
            dzdz_sph_i[GET_INDEX(l)] = dzdz_sph_i[GET_INDEX(-l)] = 0.0;
            ql2m_2 = 0.0;
        }
    }

    // m = +-(l-1)
    qlm_1 = -z * qlm_2;
    pq = qlm_1 * pk[l - 1];
    sph_i[GET_INDEX(-l + 1)] = pq * s[GET_INDEX(l - 1)];
    sph_i[GET_INDEX(+l - 1)] = pq * c[GET_INDEX(l - 1)];

    if constexpr (DO_DERIVATIVES) {
        pq *= (l - 1);
        dx_sph_i[GET_INDEX(-l + 1)] = pq * s[GET_INDEX(l - 2)];
        dy_sph_i[GET_INDEX(-l + 1)] = dx_sph_i[GET_INDEX(l - 1)] =
            pq * c[GET_INDEX(l - 2)];
        dy_sph_i[GET_INDEX(l - 1)] = -dx_sph_i[GET_INDEX(-l + 1)];

        // uses Q(l-1)(l-1) to initialize the Qlm  recursion
        ql1m_1 = qlmk[-1];
        auto pdq = pk[l - 1] * (l + l - 1) * ql1m_1;
        dz_sph_i[GET_INDEX(-l + 1)] = pdq * s[GET_INDEX(l - 1)];
        dz_sph_i[GET_INDEX(l - 1)] = pdq * c[GET_INDEX(l - 1)];

        if constexpr (DO_SECOND_DERIVATIVES) {
            pq *= (l - 2);
            if (l == 2) { // this is a special case for second derivatives
                dxdx_sph_i[GET_INDEX(l - 1)] = 0;
                dxdx_sph_i[GET_INDEX(-l + 1)] = 0;
            } else {
                dxdx_sph_i[GET_INDEX(l - 1)] = pq * c[GET_INDEX(l - 3)];
                dxdx_sph_i[GET_INDEX(-l + 1)] = pq * s[GET_INDEX(l - 3)];
            }
            dxdy_sph_i[GET_INDEX(l - 1)] = dydx_sph_i[GET_INDEX(l - 1)] =
                dydy_sph_i[GET_INDEX(-l + 1)] = -dxdx_sph_i[GET_INDEX(-l + 1)];
            dxdy_sph_i[GET_INDEX(-l + 1)] = dydx_sph_i[GET_INDEX(-l + 1)] =
                dxdx_sph_i[GET_INDEX(l - 1)];
            auto temp = -pk[l - 1] * (l - 1) *
                        qlm_2; // this is p[l-1]*q[l-1][l-1]*(2*l-1)*(l-1) =
                               // p[l-1]*(l-1)*Q_ll
            dxdz_sph_i[GET_INDEX(l - 1)] = dzdx_sph_i[GET_INDEX(l - 1)] =
                temp * c[GET_INDEX(l - 2)];
            dxdz_sph_i[GET_INDEX(-l + 1)] = dzdx_sph_i[GET_INDEX(-l + 1)] =
                temp * s[GET_INDEX(l - 2)];
            dydy_sph_i[GET_INDEX(l - 1)] = -dxdx_sph_i[GET_INDEX(l - 1)];
            dydz_sph_i[GET_INDEX(l - 1)] = dzdy_sph_i[GET_INDEX(l - 1)] =
                -dxdz_sph_i[GET_INDEX(-l + 1)];
            dydz_sph_i[GET_INDEX(-l + 1)] = dzdy_sph_i[GET_INDEX(-l + 1)] =
                dxdz_sph_i[GET_INDEX(l - 1)];
            dzdz_sph_i[GET_INDEX(l - 1)] = dzdz_sph_i[GET_INDEX(-l + 1)] = 0.0;
            ql2m_1 = 0.0;
        }
    }

    // and now do the other m's, decrementally
    for (auto m = l - 2; m > 0; --m) {
        qlm_0 = qlmk[m] * (twomz[GET_INDEX(m)] * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];
        sph_i[GET_INDEX(-m)] = pq * s[GET_INDEX(m)];
        sph_i[GET_INDEX(+m)] = pq * c[GET_INDEX(m)];

        if constexpr (DO_DERIVATIVES) {
            ql1m_0 =
                qlmk[m - l] * (twomz[GET_INDEX(m)] * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1;
            ql1m_1 = ql1m_0; // shift

            pq *= m;
            auto pqs = pq * s[GET_INDEX(m - 1)], pqc = pq * c[GET_INDEX(m - 1)];
            auto pdq = pk[m] * ql1m_2;
            auto pdqx = pdq * x;
            dx_sph_i[GET_INDEX(-m)] = (pdqx * s[GET_INDEX(m)] + pqs);
            dx_sph_i[GET_INDEX(+m)] = (pdqx * c[GET_INDEX(m)] + pqc);
            auto pdqy = pdq * y;
            dy_sph_i[GET_INDEX(-m)] = (pdqy * s[GET_INDEX(m)] + pqc);
            dy_sph_i[GET_INDEX(m)] = (pdqy * c[GET_INDEX(m)] - pqs);
            pdq = pk[m] * (l + m) * ql1m_1;
            dz_sph_i[GET_INDEX(-m)] = pdq * s[GET_INDEX(m)];
            dz_sph_i[GET_INDEX(m)] = pdq * c[GET_INDEX(m)];

            if constexpr (DO_SECOND_DERIVATIVES) {
                if (m == l - 2) {
                    // In this case, the recursion still needs to be initialized
                    // using Q(l-2)(l-2)
                    ql2m_0 = qlmk[-l - 1];
                } else {
                    // Recursion
                    ql2m_0 = qlmk[m - 2 * l + 1] *
                             (twomz[GET_INDEX(m)] * ql2m_1 + rxy * ql2m_2);
                }

                pq /= m;
                auto pql1m_1 =
                    pk[m] * ql1m_2; // Note the index discrepancy: ql1m_1
                                    // was already shifted above to ql1m_2
                auto pql2m_2 = pk[m] * ql2m_2;
                auto pql2m_0 = pk[m] * ql2m_0;
                auto pql1m_0 =
                    pk[m] * ql1m_1; // Note the index discrepancy: ql1m_0
                                    // was already shifted above to ql1m_1
                auto pql2m_1 = pk[m] * ql2m_1;

                // Diagonal hessian terms
                T mmpqc2 = 0.0;
                T mmpqs2 = 0.0;

                if (m != 1) {
                    mmpqc2 = m * (m - 1) * pq * c[GET_INDEX(m - 2)];
                    mmpqs2 = m * (m - 1) * pq * s[GET_INDEX(m - 2)];
                }

                dxdx_sph_i[GET_INDEX(m)] =
                    pql1m_1 * c[GET_INDEX(m)] + x2 * pql2m_2 * c[GET_INDEX(m)] +
                    2 * m * x * pql1m_1 * c[GET_INDEX(m - 1)] + mmpqc2;
                dxdx_sph_i[GET_INDEX(-m)] =
                    pql1m_1 * s[GET_INDEX(m)] + x2 * pql2m_2 * s[GET_INDEX(m)] +
                    2 * m * x * pql1m_1 * s[GET_INDEX(m - 1)] + mmpqs2;
                dydy_sph_i[GET_INDEX(m)] =
                    pql1m_1 * c[GET_INDEX(m)] + y2 * pql2m_2 * c[GET_INDEX(m)] -
                    2 * m * y * pql1m_1 * s[GET_INDEX(m - 1)] - mmpqc2;
                dydy_sph_i[GET_INDEX(-m)] =
                    pql1m_1 * s[GET_INDEX(m)] + y2 * pql2m_2 * s[GET_INDEX(m)] +
                    2 * m * y * pql1m_1 * c[GET_INDEX(m - 1)] - mmpqs2;
                dzdz_sph_i[GET_INDEX(m)] =
                    (l + m) * (l + m - 1) * pql2m_0 * c[GET_INDEX(m)];
                dzdz_sph_i[GET_INDEX(-m)] =
                    (l + m) * (l + m - 1) * pql2m_0 * s[GET_INDEX(m)];

                // Off-diagonal terms. Note that these are symmetric
                dxdy_sph_i[GET_INDEX(m)] = dydx_sph_i[GET_INDEX(m)] =
                    xy * pql2m_2 * c[GET_INDEX(m)] +
                    y * pql1m_1 * m * c[GET_INDEX(m - 1)] -
                    x * pql1m_1 * m * s[GET_INDEX(m - 1)] - mmpqs2;
                dxdy_sph_i[GET_INDEX(-m)] = dydx_sph_i[GET_INDEX(-m)] =
                    xy * pql2m_2 * s[GET_INDEX(m)] +
                    y * pql1m_1 * m * s[GET_INDEX(m - 1)] +
                    x * pql1m_1 * m * c[GET_INDEX(m - 1)] + mmpqc2;
                dxdz_sph_i[GET_INDEX(m)] = dzdx_sph_i[GET_INDEX(m)] =
                    x * (l + m) * pql2m_1 * c[GET_INDEX(m)] +
                    (l + m) * pql1m_0 * m * c[GET_INDEX(m - 1)];
                dxdz_sph_i[GET_INDEX(-m)] = dzdx_sph_i[GET_INDEX(-m)] =
                    x * (l + m) * pql2m_1 * s[GET_INDEX(m)] +
                    (l + m) * pql1m_0 * m * s[GET_INDEX(m - 1)];
                dydz_sph_i[GET_INDEX(m)] = dzdy_sph_i[GET_INDEX(m)] =
                    y * (l + m) * pql2m_1 * c[GET_INDEX(m)] -
                    (l + m) * pql1m_0 * m * s[GET_INDEX(m - 1)];
                dydz_sph_i[GET_INDEX(-m)] = dzdy_sph_i[GET_INDEX(-m)] =
                    y * (l + m) * pql2m_1 * s[GET_INDEX(m)] +
                    (l + m) * pql1m_0 * m * c[GET_INDEX(m - 1)];

                ql2m_2 = ql2m_1;
                ql2m_1 = ql2m_0; // shift at the end because we need all three
                                 // at the same time
            }
        }
    }

    // m=0 is also a special case
    qlm_0 = qlmk[0] * (twomz[GET_INDEX(0)] * qlm_1 + rxy * qlm_2);
    sph_i[GET_INDEX(0)] = qlm_0 * pk[0];

    if constexpr (DO_DERIVATIVES) {
        ql1m_0 = qlmk[-l] * (twomz[GET_INDEX(0)] * ql1m_1 + rxy * ql1m_2);
        // derivatives
        dx_sph_i[GET_INDEX(0)] = pk[0] * x * ql1m_1;
        dy_sph_i[GET_INDEX(0)] = pk[0] * y * ql1m_1;
        dz_sph_i[GET_INDEX(0)] = pk[0] * l * ql1m_0;

        if constexpr (DO_SECOND_DERIVATIVES) {
            if (l == 2) {
                // special case: recursion is not initialized yet
                ql2m_0 = qlmk[-2 * l + 1];
            } else {
                ql2m_0 = qlmk[-2 * l + 1] *
                         (twomz[GET_INDEX(0)] * ql2m_1 + rxy * ql2m_2);
            }

            auto pql1m_1 = pk[0] * ql1m_1;
            auto pql2m_2 = pk[0] * ql2m_2;
            auto pql2m_0 = pk[0] * ql2m_0;
            auto pql2m_1 = pk[0] * ql2m_1;

            // diagonal
            dxdx_sph_i[GET_INDEX(0)] = pql1m_1 + x2 * pql2m_2;
            dydy_sph_i[GET_INDEX(0)] = pql1m_1 + y2 * pql2m_2;
            dzdz_sph_i[GET_INDEX(0)] = (l) * (l - 1) * pql2m_0;

            // off-diagonal (symmetric)
            dxdy_sph_i[GET_INDEX(0)] = dydx_sph_i[GET_INDEX(0)] = xy * pql2m_2;
            dxdz_sph_i[GET_INDEX(0)] = dzdx_sph_i[GET_INDEX(0)] =
                x * l * pql2m_1;
            dydz_sph_i[GET_INDEX(0)] = dzdy_sph_i[GET_INDEX(0)] =
                y * l * pql2m_1;
        }
    }
}

template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          int HARDCODED_LMAX, int TEMPLATED_L,
          int (*GET_INDEX)(int) = dummy_idx>
CUDA_DEVICE_PREFIX static inline void generic_sph_l_channel_templated(
    [[maybe_unused]] T x, // these might be unused for low LMAX. not worth a
                          // full separate implementation
    [[maybe_unused]] T y, [[maybe_unused]] T z, [[maybe_unused]] T rxy,
    const T *pk, const T *qlmk, T *c, T *s, T *twomz, T *sph_i,
    [[maybe_unused]] T *dx_sph_i, [[maybe_unused]] T *dy_sph_i,
    [[maybe_unused]] T *dz_sph_i, [[maybe_unused]] T *dxdx_sph_i,
    [[maybe_unused]] T *dxdy_sph_i, [[maybe_unused]] T *dxdz_sph_i,
    [[maybe_unused]] T *dydx_sph_i, [[maybe_unused]] T *dydy_sph_i,
    [[maybe_unused]] T *dydz_sph_i, [[maybe_unused]] T *dzdx_sph_i,
    [[maybe_unused]] T *dzdy_sph_i, [[maybe_unused]] T *dzdz_sph_i) {

    generic_sph_l_channel<T, DO_DERIVATIVES, DO_SECOND_DERIVATIVES,
                          HARDCODED_LMAX>(
        TEMPLATED_L, x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
        dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i, dydx_sph_i,
        dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i, dzdz_sph_i);
}

template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          bool NORMALIZED, int HARDCODED_LMAX, int TEMPLATED_LMAX>
static inline void
generic_sph_sample(const T *xyz_i, T *sph_i, [[maybe_unused]] T *dsph_i,
                   [[maybe_unused]] T *ddsph_i, int l_max, int size_y,
                   const T *pylm, const T *pqlm, T *c, T *s, T *twomz) {
    /*
    This is a low-level function that combines all the pieces to evaluate the
    sph for a single sample. It calls both the hardcoded macros and the generic
    l-channel calculator, as well as the sine and cosine terms that are combined
    with the Qlm, that are needed in the generic calculator. There is a lot of
    pointer algebra used to address the correct part of the various arrays, that
    turned out to be more efficient than explicit indexing in early tests.

    The parameters correspond to those described in generic_sph_l_channel.
    */
    static_assert(!(DO_SECOND_DERIVATIVES && HARDCODED_LMAX > 1),
                  "Hardcoded second derivatives are not implemented for l>1.");

    [[maybe_unused]] T ir =
        0.0; // storage for computing 1/r, which is reused when NORMALIZED=true

    // pointers for first derivatives
    [[maybe_unused]] T *dx_sph_i = nullptr;
    [[maybe_unused]] T *dy_sph_i = nullptr;
    [[maybe_unused]] T *dz_sph_i = nullptr;

    // pointers for second derivatives
    [[maybe_unused]] T *dxdx_sph_i = nullptr;
    [[maybe_unused]] T *dxdy_sph_i = nullptr;
    [[maybe_unused]] T *dxdz_sph_i = nullptr;
    [[maybe_unused]] T *dydx_sph_i = nullptr;
    [[maybe_unused]] T *dydy_sph_i = nullptr;
    [[maybe_unused]] T *dydz_sph_i = nullptr;
    [[maybe_unused]] T *dzdx_sph_i = nullptr;
    [[maybe_unused]] T *dzdy_sph_i = nullptr;
    [[maybe_unused]] T *dzdz_sph_i = nullptr;

    /* k is a utility index to traverse lm arrays. we store sph in
    a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
    so we often write a nested loop on l and m and track where we
    got by incrementing a separate index k. */
    int k = 0;

    auto x = xyz_i[0];
    auto y = xyz_i[1];
    auto z = xyz_i[2];
    [[maybe_unused]] auto x2 = x * x;
    [[maybe_unused]] auto y2 = y * y;
    [[maybe_unused]] auto z2 = z * z;
    if constexpr (NORMALIZED) {
        ir = 1 / std::sqrt(x2 + y2 + z2);
        x *= ir;
        y *= ir;
        z *= ir;
        x2 = x * x;
        y2 = y * y;
        z2 = z * z;
    }
    auto rxy = x2 + y2;

    // these are the hard-coded, low-lmax sph
    HARDCODED_SPH_MACRO(HARDCODED_LMAX, x, y, z, x2, y2, z2, sph_i,
                        DUMMY_SPH_IDX);

    if constexpr (DO_DERIVATIVES) {
        // updates the pointer to the derivative storage
        dx_sph_i = dsph_i;
        dy_sph_i = dx_sph_i + size_y;
        dz_sph_i = dy_sph_i + size_y;

        // these are the hard-coded, low-lmax dsph
        HARDCODED_SPH_DERIVATIVE_MACRO(HARDCODED_LMAX, x, y, z, x2, y2, z2,
                                       sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
                                       DUMMY_SPH_IDX);
    }

    if constexpr (DO_SECOND_DERIVATIVES) {
        // set each double derivative pointer to the appropriate place
        dxdx_sph_i = ddsph_i;
        dxdy_sph_i = dxdx_sph_i + size_y;
        dxdz_sph_i = dxdy_sph_i + size_y;
        dydx_sph_i = dxdz_sph_i + size_y;
        dydy_sph_i = dydx_sph_i + size_y;
        dydz_sph_i = dydy_sph_i + size_y;
        dzdx_sph_i = dydz_sph_i + size_y;
        dzdy_sph_i = dzdx_sph_i + size_y;
        dzdz_sph_i = dzdy_sph_i + size_y;

        // these are the hard-coded, low-lmax ddsph
        HARDCODED_SPH_SECOND_DERIVATIVE_MACRO(
            HARDCODED_LMAX, sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
            dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
            dzdz_sph_i, DUMMY_SPH_IDX);
    }

    /* These are scaled version of cos(m phi) and sin(m phi).
        Basically, these are cos and sin multiplied by r_xy^m,
        so that they are just plain polynomials of x,y,z.    */
    // help the compiler unroll the first part of the loop
    int m = 0;
    auto twoz = z + z; // twomz actually holds 2*(m+1)*z
    twomz[0] = twoz;
    // also initialize the sine and cosine, even if these never change
    c[0] = 1.0;
    s[0] = 0.0;
    for (m = 1; m < HARDCODED_LMAX + 1;  // can we change this to TEMPLATED_LMAX somehow?
         ++m) { // allow unrolling of the static part of the loop
        c[m] = c[m - 1] * x - s[m - 1] * y;
        s[m] = c[m - 1] * y + s[m - 1] * x;
        twomz[m] = twomz[m - 1] + twoz;
    }
    for (; m < l_max + 1; m++) {
        c[m] = c[m - 1] * x - s[m - 1] * y;
        s[m] = c[m - 1] * y + s[m - 1] * x;
        twomz[m] = twomz[m - 1] + twoz;
    }

    /* compute recursively the "Cartesian" associated Legendre polynomials Qlm.
        Qlm is defined as r^l/r_xy^m P_lm, and is a polynomial of x,y,z.
        These are computed with a recursive expression.

        Also assembles the (Cartesian) sph by combining Qlm and
        sine/cosine phi-dependent factors. we use pointer
        arithmetics to make sure spk_i always points at the
        beginning of the appropriate memory segment.   */

    // main loop!
    // k points at Q[l,0]; sph_i at Y[l,0] (mid-way through each l chunk)
    k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;
    sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);

    if constexpr (DO_DERIVATIVES) {
        dx_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dy_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dz_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
    }

    if constexpr (DO_SECOND_DERIVATIVES) {
        dxdx_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dxdy_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dxdz_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dydx_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dydy_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dydz_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dzdx_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dzdy_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dzdz_sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
    }

    auto pk = pylm + k;
    auto qlmk = pqlm + k; // starts at HARDCODED_LMAX+1
    for (int l = HARDCODED_LMAX + 1; l < l_max + 1; l++) {
        if (l <= TEMPLATED_LMAX) {
            // call templated version
            // generic_sph_l_channel_templated<T, DO_DERIVATIVES,
            // DO_SECOND_DERIVATIVES,
            //                       HARDCODED_LMAX, l>(
            //     x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
            //     dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
            //     dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
            //     dzdz_sph_i);
            if (l == 2) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 2>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 3) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 3>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 4) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 4>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 5) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 5>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 6) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 6>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 7) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 7>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 8) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 8>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 9) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 9>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
            if (l == 10) {
                generic_sph_l_channel_templated<T, DO_DERIVATIVES,
                                                DO_SECOND_DERIVATIVES,
                                                HARDCODED_LMAX, 10>(
                    x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                    dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                    dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                    dzdz_sph_i);
            }
        } else {
            generic_sph_l_channel<T, DO_DERIVATIVES, DO_SECOND_DERIVATIVES,
                                  HARDCODED_LMAX>(
                l, x, y, z, rxy, pk, qlmk, c, s, twomz, sph_i, dx_sph_i,
                dy_sph_i, dz_sph_i, dxdx_sph_i, dxdy_sph_i, dxdz_sph_i,
                dydx_sph_i, dydy_sph_i, dydz_sph_i, dzdx_sph_i, dzdy_sph_i,
                dzdz_sph_i);
        }

        // shift pointers & indexes to the next l block
        qlmk += l + 1;
        pk += l + 1;
        sph_i += 2 * l + 2;

        if constexpr (DO_DERIVATIVES) {
            dx_sph_i += 2 * l + 2;
            dy_sph_i += 2 * l + 2;
            dz_sph_i += 2 * l + 2;
        }

        if constexpr (DO_SECOND_DERIVATIVES) {
            dxdx_sph_i += 2 * l + 2;
            dxdy_sph_i += 2 * l + 2;
            dxdz_sph_i += 2 * l + 2;
            dydx_sph_i += 2 * l + 2;
            dydy_sph_i += 2 * l + 2;
            dydz_sph_i += 2 * l + 2;
            dzdx_sph_i += 2 * l + 2;
            dzdy_sph_i += 2 * l + 2;
            dzdz_sph_i += 2 * l + 2;
        }
    }

    if constexpr (DO_DERIVATIVES && NORMALIZED) {
        // corrects derivatives for normalization
        dx_sph_i = dsph_i;
        dy_sph_i = dx_sph_i + size_y;
        dz_sph_i = dy_sph_i + size_y;

        if constexpr (DO_SECOND_DERIVATIVES) {
            // set each second derivative pointer to the appropriate place
            dxdx_sph_i = ddsph_i;
            dxdy_sph_i = dxdx_sph_i + size_y;
            dxdz_sph_i = dxdy_sph_i + size_y;
            dydx_sph_i = dxdz_sph_i + size_y;
            dydy_sph_i = dydx_sph_i + size_y;
            dydz_sph_i = dydy_sph_i + size_y;
            dzdx_sph_i = dydz_sph_i + size_y;
            dzdy_sph_i = dzdx_sph_i + size_y;
            dzdz_sph_i = dzdy_sph_i + size_y;
        }

        for (k = 0; k < size_y; ++k) {
            auto tmp = (dx_sph_i[k] * x + dy_sph_i[k] * y + dz_sph_i[k] * z);

            if constexpr (DO_SECOND_DERIVATIVES) {
                // correct second derivatives for normalization.
                // We do it before the first derivatives because we need the
                // first derivatives in their non-normalized form
                auto irsq = ir * ir;
                auto tmpx =
                    x * dxdx_sph_i[k] + y * dydx_sph_i[k] + z * dzdx_sph_i[k];
                auto tmpy =
                    x * dxdy_sph_i[k] + y * dydy_sph_i[k] + z * dydz_sph_i[k];
                auto tmpz =
                    x * dxdz_sph_i[k] + y * dydz_sph_i[k] + z * dzdz_sph_i[k];
                auto tmp2 = x * x * dxdx_sph_i[k] + y * y * dydy_sph_i[k] +
                            z * z * dzdz_sph_i[k] + 2 * x * y * dxdy_sph_i[k] +
                            2 * x * z * dxdz_sph_i[k] +
                            2 * y * z * dydz_sph_i[k];
                dxdx_sph_i[k] =
                    (-2 * x * tmpx + dxdx_sph_i[k] + 3 * x * x * tmp - tmp -
                     2 * x * dx_sph_i[k] + x * x * tmp2) *
                    irsq;
                dydy_sph_i[k] =
                    (-2 * y * tmpy + dydy_sph_i[k] + 3 * y * y * tmp - tmp -
                     2 * y * dy_sph_i[k] + y * y * tmp2) *
                    irsq;
                dzdz_sph_i[k] =
                    (-2 * z * tmpz + dzdz_sph_i[k] + 3 * z * z * tmp - tmp -
                     2 * z * dz_sph_i[k] + z * z * tmp2) *
                    irsq;
                dxdy_sph_i[k] = dydx_sph_i[k] =
                    (-x * tmpy - y * tmpx + dxdy_sph_i[k] + 3 * x * y * tmp -
                     x * dy_sph_i[k] - y * dx_sph_i[k] + x * y * tmp2) *
                    irsq;
                dxdz_sph_i[k] = dzdx_sph_i[k] =
                    (-x * tmpz - z * tmpx + dxdz_sph_i[k] + 3 * x * z * tmp -
                     x * dz_sph_i[k] - z * dx_sph_i[k] + x * z * tmp2) *
                    irsq;
                dzdy_sph_i[k] = dydz_sph_i[k] =
                    (-z * tmpy - y * tmpz + dzdy_sph_i[k] + 3 * y * z * tmp -
                     z * dy_sph_i[k] - y * dz_sph_i[k] + y * z * tmp2) *
                    irsq;
            }

            // correct first derivatives for normalization
            dx_sph_i[k] = (dx_sph_i[k] - x * tmp) * ir;
            dy_sph_i[k] = (dy_sph_i[k] - y * tmp) * ir;
            dz_sph_i[k] = (dz_sph_i[k] - z * tmp) * ir;
        }
    }
}

template <typename T, bool DO_DERIVATIVES, bool DO_SECOND_DERIVATIVES,
          bool NORMALIZED, int HARDCODED_LMAX, int TEMPLATED_LMAX>
void generic_sph(const T *xyz, T *sph, [[maybe_unused]] T *dsph,
                 [[maybe_unused]] T *ddsph, size_t n_samples, int l_max,
                 const T *prefactors, T *buffers) {
    /*
        Implementation of the general Ylm calculator case. Starts at
       HARDCODED_LMAX and uses hard-coding before that.

        Some general implementation remarks:
        - we use an alternative iteration for the Qlm that avoids computing the
          low-l section
        - there is OMP parallelism threading over the samples (there is probably
       lots to optimize on this front)
        - we compute at the same time Qlm and the corresponding Ylm, to reuse
          more of the pieces and stay local in memory. we use `if constexpr`
          to avoid runtime branching in the DO_DERIVATIVES=true/false cases
        - we explicitly split the loops in a fixed-length (depending on the
          template parameter HARDCODED_LMAX) and a variable lenght one, so that
          the compiler can choose to unroll if it makes sense
        - there's a bit of pointer gymnastics because of the contiguous storage
          of the irregularly-shaped Q[l,m] and Y[l,m] arrays

        Template parameters:
        typename T: float type (e.g. single/double precision)
        bool DO_DERIVATIVES: should we evaluate the derivatives?
        bool DO_SECOND_DERIVATIVES: should we evaluate the second derivatives?
        bool NORMALIZED: should we normalize the input positions?
        int HARDCODED_LMAX: which lmax value will be computed

        Actual parameters:
        const T *xyz: a T array containing th n_samplex*3 x,y,z coordinates of
       multiple 3D points T *sph: pointer to the storage location for the Ylm
       (stored as l,m= (0,0),(1,-1),(1,0),(1,1),...
        [[maybe_unused]] T *dsph : pointer to the storage location for the
       dYlm/dx,dy,dz. stored as for sph_i, with three consecutive blocks
       associated to d/dx,d/dy,d/dz
        [[maybe_unused]] T *dsph : pointer to the storage location for the
       second derivatives. stored as for sph_i, with nine consecutive blocks
       associated to the possible second derivative combinations size_t
       n_samples: number of samples that have to be computed int l_max: maximum
       l to compute prefactors: pointer to an array that contains the prefactors
       used for Ylm and Qlm calculation buffers: buffer space to compute cosine,
       sine and 2*m*z terms
    */

    // implementation assumes to use hardcoded expressions for at least l=0,1
    static_assert(HARDCODED_LMAX >= 1,
                  "Cannot call the generic Ylm calculator for l<=1.");

    const auto size_y = (l_max + 1) * (l_max + 1);     // size of Ylm blocks
    const auto size_q = (l_max + 1) * (l_max + 2) / 2; // size of Qlm blocks
    const T *qlmfactors =
        prefactors + size_q; // the coeffs. used to compute Qlm are just stored
                             // contiguously after the Ylm prefactors

#pragma omp parallel
    {
        auto c = buffers + omp_get_thread_num() * size_q * 3;
        auto s = c + size_q;
        auto twomz = s + size_q;
        // ^^^ thread-local storage arrays for terms corresponding to (scaled)
        // cosine and sine of the azimuth, and 2mz

        // pointers to the sections of the output arrays that hold Ylm and
        // derivatives for a given point
        T *sph_i = nullptr;
        T *dsph_i = nullptr;
        T *ddsph_i = nullptr;

#pragma omp for
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            auto xyz_i = xyz + i_sample * 3;
            // pointer to the segment that should store the i_sample sph
            sph_i = sph + i_sample * size_y;
            if constexpr (DO_DERIVATIVES) {
                // updates the pointer to the derivative storage
                dsph_i = dsph + i_sample * 3 * size_y;
            }
            if constexpr (DO_SECOND_DERIVATIVES) {
                // updates the pointer to the second derivative storage
                ddsph_i = ddsph + i_sample * 9 * size_y;
            }

            generic_sph_sample<T, DO_DERIVATIVES, DO_SECOND_DERIVATIVES,
                               NORMALIZED, HARDCODED_LMAX, TEMPLATED_LMAX>(
                xyz_i, sph_i, dsph_i, ddsph_i, l_max, size_y, prefactors,
                qlmfactors, c, s, twomz);
        }
    }
}

#endif
