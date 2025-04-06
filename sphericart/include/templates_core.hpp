#ifndef SPHERICART_TEMPLATES_CORE_HPP
#define SPHERICART_TEMPLATES_CORE_HPP

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

// a SPH_IDX that does nothing
#define DUMMY_SPH_IDX

int inline CUDA_DEVICE_PREFIX dummy_idx(int i) { return i; }

/** Computes the sph and their derivatives for a given Cartesian point and a
 * given l. The template implementation supports different floating poitn types
 * T, determines whether to compute derivatives (DO_DERIVATIVES), assumes that
 * l is greater than HARDCODED_LMAX. GET_INDEX is a function that might allow
 * to map differently the indices in the spherical harmonics (used in the CUDA
 * implementation).
 */
template <
    typename T,
    bool DO_DERIVATIVES,
    bool DO_SECOND_DERIVATIVES,
    int HARDCODED_LMAX,
    int (*GET_INDEX)(int) = dummy_idx>
CUDA_DEVICE_PREFIX static inline void generic_sph_l_channel(
    int l,
    [[maybe_unused]] T x, // these might be unused for low LMAX. not worth a
                          // full separate implementation
    [[maybe_unused]] T y,
    [[maybe_unused]] T z,
    [[maybe_unused]] T rxy,
    const T* pk,
    const T* qlmk,
    T* c,
    T* s,
    T* twomz,
    T* sph_i,
    [[maybe_unused]] T* dx_sph_i,
    [[maybe_unused]] T* dy_sph_i,
    [[maybe_unused]] T* dz_sph_i,
    [[maybe_unused]] T* dxdx_sph_i,
    [[maybe_unused]] T* dxdy_sph_i,
    [[maybe_unused]] T* dxdz_sph_i,
    [[maybe_unused]] T* dydx_sph_i,
    [[maybe_unused]] T* dydy_sph_i,
    [[maybe_unused]] T* dydz_sph_i,
    [[maybe_unused]] T* dzdx_sph_i,
    [[maybe_unused]] T* dzdy_sph_i,
    [[maybe_unused]] T* dzdz_sph_i
) {
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
        "Cannot calculate second derivatives without first derivatives"
    );

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
        dy_sph_i[GET_INDEX(-l)] = dx_sph_i[GET_INDEX(l)] = pq * c[GET_INDEX(l - 1)];
        dy_sph_i[GET_INDEX(l)] = -dx_sph_i[GET_INDEX(-l)];
        dz_sph_i[GET_INDEX(-l)] = 0;
        dz_sph_i[GET_INDEX(l)] = 0;
        ql1m_2 = 0;

        if constexpr (DO_SECOND_DERIVATIVES) {
            pq *= (l - 1);
            dxdx_sph_i[GET_INDEX(l)] = pq * c[GET_INDEX(l - 2)];
            dxdx_sph_i[GET_INDEX(-l)] = pq * s[GET_INDEX(l - 2)];
            dxdy_sph_i[GET_INDEX(l)] = dydx_sph_i[GET_INDEX(l)] = dydy_sph_i[GET_INDEX(-l)] =
                -dxdx_sph_i[GET_INDEX(-l)];
            dxdy_sph_i[GET_INDEX(-l)] = dydx_sph_i[GET_INDEX(-l)] = dxdx_sph_i[GET_INDEX(l)];
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
        dy_sph_i[GET_INDEX(-l + 1)] = dx_sph_i[GET_INDEX(l - 1)] = pq * c[GET_INDEX(l - 2)];
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
            auto temp = -pk[l - 1] * (l - 1) * qlm_2; // this is p[l-1]*q[l-1][l-1]*(2*l-1)*(l-1) =
                                                      // p[l-1]*(l-1)*Q_ll
            dxdz_sph_i[GET_INDEX(l - 1)] = dzdx_sph_i[GET_INDEX(l - 1)] = temp * c[GET_INDEX(l - 2)];
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
    for (auto m = l - 2; m > HARDCODED_LMAX - 1; --m) {
        qlm_0 = qlmk[m] * (twomz[GET_INDEX(m)] * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift
        pq = qlm_0 * pk[m];
        sph_i[GET_INDEX(-m)] = pq * s[GET_INDEX(m)];
        sph_i[GET_INDEX(+m)] = pq * c[GET_INDEX(m)];

        if constexpr (DO_DERIVATIVES) {
            ql1m_0 = qlmk[m - l] * (twomz[GET_INDEX(m)] * ql1m_1 + rxy * ql1m_2);
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
                    ql2m_0 = qlmk[m - 2 * l + 1] * (twomz[GET_INDEX(m)] * ql2m_1 + rxy * ql2m_2);
                }

                pq /= m;
                auto pql1m_1 = pk[m] * ql1m_2; // Note the index discrepancy: ql1m_1
                                               // was already shifted above to ql1m_2
                auto pql2m_2 = pk[m] * ql2m_2;
                auto pql2m_0 = pk[m] * ql2m_0;
                auto pql1m_0 = pk[m] * ql1m_1; // Note the index discrepancy: ql1m_0
                                               // was already shifted above to ql1m_1
                auto pql2m_1 = pk[m] * ql2m_1;

                // Diagonal hessian terms
                T mmpqc2 = 0.0;
                T mmpqs2 = 0.0;
                if (m != 1) {
                    mmpqc2 = m * (m - 1) * pq * c[GET_INDEX(m - 2)];
                    mmpqs2 = m * (m - 1) * pq * s[GET_INDEX(m - 2)];
                }

                dxdx_sph_i[GET_INDEX(m)] = pql1m_1 * c[GET_INDEX(m)] +
                                           x2 * pql2m_2 * c[GET_INDEX(m)] +
                                           2 * m * x * pql1m_1 * c[GET_INDEX(m - 1)] + mmpqc2;
                dxdx_sph_i[GET_INDEX(-m)] = pql1m_1 * s[GET_INDEX(m)] +
                                            x2 * pql2m_2 * s[GET_INDEX(m)] +
                                            2 * m * x * pql1m_1 * s[GET_INDEX(m - 1)] + mmpqs2;
                dydy_sph_i[GET_INDEX(m)] = pql1m_1 * c[GET_INDEX(m)] +
                                           y2 * pql2m_2 * c[GET_INDEX(m)] -
                                           2 * m * y * pql1m_1 * s[GET_INDEX(m - 1)] - mmpqc2;
                dydy_sph_i[GET_INDEX(-m)] = pql1m_1 * s[GET_INDEX(m)] +
                                            y2 * pql2m_2 * s[GET_INDEX(m)] +
                                            2 * m * y * pql1m_1 * c[GET_INDEX(m - 1)] - mmpqs2;
                dzdz_sph_i[GET_INDEX(m)] = (l + m) * (l + m - 1) * pql2m_0 * c[GET_INDEX(m)];
                dzdz_sph_i[GET_INDEX(-m)] = (l + m) * (l + m - 1) * pql2m_0 * s[GET_INDEX(m)];

                // Off-diagonal terms. Note that these are symmetric
                dxdy_sph_i[GET_INDEX(m)] = dydx_sph_i[GET_INDEX(m)] =
                    xy * pql2m_2 * c[GET_INDEX(m)] + y * pql1m_1 * m * c[GET_INDEX(m - 1)] -
                    x * pql1m_1 * m * s[GET_INDEX(m - 1)] - mmpqs2;
                dxdy_sph_i[GET_INDEX(-m)] = dydx_sph_i[GET_INDEX(-m)] =
                    xy * pql2m_2 * s[GET_INDEX(m)] + y * pql1m_1 * m * s[GET_INDEX(m - 1)] +
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
    for (auto m = HARDCODED_LMAX - 1; m > 0; --m) {
        qlm_0 = qlmk[m] * (twomz[GET_INDEX(m)] * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];
        sph_i[GET_INDEX(-m)] = pq * s[GET_INDEX(m)];
        sph_i[GET_INDEX(+m)] = pq * c[GET_INDEX(m)];

        if constexpr (DO_DERIVATIVES) {
            ql1m_0 = qlmk[m - l] * (twomz[GET_INDEX(m)] * ql1m_1 + rxy * ql1m_2);
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
                    ql2m_0 = qlmk[m - 2 * l + 1] * (twomz[GET_INDEX(m)] * ql2m_1 + rxy * ql2m_2);
                }

                pq /= m;
                auto pql1m_1 = pk[m] * ql1m_2; // Note the index discrepancy: ql1m_1
                                               // was already shifted above to ql1m_2
                auto pql2m_2 = pk[m] * ql2m_2;
                auto pql2m_0 = pk[m] * ql2m_0;
                auto pql1m_0 = pk[m] * ql1m_1; // Note the index discrepancy: ql1m_0
                                               // was already shifted above to ql1m_1
                auto pql2m_1 = pk[m] * ql2m_1;

                // Diagonal hessian terms
                T mmpqc2 = 0.0;
                T mmpqs2 = 0.0;

                if (m != 1) {
                    mmpqc2 = m * (m - 1) * pq * c[GET_INDEX(m - 2)];
                    mmpqs2 = m * (m - 1) * pq * s[GET_INDEX(m - 2)];
                }

                dxdx_sph_i[GET_INDEX(m)] = pql1m_1 * c[GET_INDEX(m)] +
                                           x2 * pql2m_2 * c[GET_INDEX(m)] +
                                           2 * m * x * pql1m_1 * c[GET_INDEX(m - 1)] + mmpqc2;
                dxdx_sph_i[GET_INDEX(-m)] = pql1m_1 * s[GET_INDEX(m)] +
                                            x2 * pql2m_2 * s[GET_INDEX(m)] +
                                            2 * m * x * pql1m_1 * s[GET_INDEX(m - 1)] + mmpqs2;
                dydy_sph_i[GET_INDEX(m)] = pql1m_1 * c[GET_INDEX(m)] +
                                           y2 * pql2m_2 * c[GET_INDEX(m)] -
                                           2 * m * y * pql1m_1 * s[GET_INDEX(m - 1)] - mmpqc2;
                dydy_sph_i[GET_INDEX(-m)] = pql1m_1 * s[GET_INDEX(m)] +
                                            y2 * pql2m_2 * s[GET_INDEX(m)] +
                                            2 * m * y * pql1m_1 * c[GET_INDEX(m - 1)] - mmpqs2;
                dzdz_sph_i[GET_INDEX(m)] = (l + m) * (l + m - 1) * pql2m_0 * c[GET_INDEX(m)];
                dzdz_sph_i[GET_INDEX(-m)] = (l + m) * (l + m - 1) * pql2m_0 * s[GET_INDEX(m)];

                // Off-diagonal terms. Note that these are symmetric
                dxdy_sph_i[GET_INDEX(m)] = dydx_sph_i[GET_INDEX(m)] =
                    xy * pql2m_2 * c[GET_INDEX(m)] + y * pql1m_1 * m * c[GET_INDEX(m - 1)] -
                    x * pql1m_1 * m * s[GET_INDEX(m - 1)] - mmpqs2;
                dxdy_sph_i[GET_INDEX(-m)] = dydx_sph_i[GET_INDEX(-m)] =
                    xy * pql2m_2 * s[GET_INDEX(m)] + y * pql1m_1 * m * s[GET_INDEX(m - 1)] +
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
                ql2m_0 = qlmk[-2 * l + 1] * (twomz[GET_INDEX(0)] * ql2m_1 + rxy * ql2m_2);
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
            dxdz_sph_i[GET_INDEX(0)] = dzdx_sph_i[GET_INDEX(0)] = x * l * pql2m_1;
            dydz_sph_i[GET_INDEX(0)] = dzdy_sph_i[GET_INDEX(0)] = y * l * pql2m_1;
        }
    }
}

#endif