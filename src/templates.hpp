#ifndef SPHERICART_TEMPLATES_HPP
#define SPHERICART_TEMPLATES_HPP

/*
    Template implementation of Cartesian Ylm calculators.

    The template functions use compile-time `if constexpr()` constructs to
    implement calculators for spherical harmonics that can handle different
    type of calls, e.g. with or without derivative calculations, and with
    different numbers of terms computed with hard-coded expressions.
*/

#include <vector>

#include "macros.hpp"

template <int HARDCODED_LMAX>
inline void hardcoded_sph_template(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
    static_assert(HARDCODED_LMAX <= SPHERICART_LMAX_HARDCODED, "Computing hardcoded sph beyond what is currently implemented.");

    COMPUTE_SPH_L0(sph_i);

    if constexpr (HARDCODED_LMAX > 0) {
        COMPUTE_SPH_L1(x, y, z, sph_i);
    }

    if constexpr (HARDCODED_LMAX > 1) {
        COMPUTE_SPH_L2(x, y, z, x2, y2, z2, sph_i);
    }

    if constexpr (HARDCODED_LMAX > 2) {
        COMPUTE_SPH_L3(x, y, z, x2, y2, z2, sph_i);
    }

    if constexpr (HARDCODED_LMAX > 3) {
        COMPUTE_SPH_L4(x, y, z, x2, y2, z2, sph_i);
    }

    if constexpr (HARDCODED_LMAX > 4) {
        COMPUTE_SPH_L5(x, y, z, x2, y2, z2, sph_i);
    }

    if constexpr (HARDCODED_LMAX > 5) {
        COMPUTE_SPH_L6(x, y, z, x2, y2, z2, sph_i);
    }
}

template <int HARDCODED_LMAX>
inline void hardcoded_sph_derivative_template(
    double x,
    double y,
    double z,
    double x2,
    double y2,
    double z2,
    double *sph_i,
    double *dxsph_i,
    double *dysph_i,
    double *dzsph_i
) {

    /*
        Combines the macro hard-coded dYlm/d(x,y,z) calculators to get all the terms
        up to HC_LMAX.  This templated version evaluates the ifs at compile time
        avoiding unnecessary in-loop branching.
    */

    COMPUTE_SPH_DERIVATIVE_L0(sph_i, dxsph_i, dysph_i, dzsph_i);

    if constexpr (HARDCODED_LMAX > 0) {
        COMPUTE_SPH_DERIVATIVE_L1(sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    if constexpr (HARDCODED_LMAX > 1) {
        COMPUTE_SPH_DERIVATIVE_L2(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    if constexpr (HARDCODED_LMAX > 2) {
        COMPUTE_SPH_DERIVATIVE_L3(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    if constexpr (HARDCODED_LMAX > 3) {
        COMPUTE_SPH_DERIVATIVE_L4(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    if constexpr (HARDCODED_LMAX > 4) {
        COMPUTE_SPH_DERIVATIVE_L5(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    if constexpr (HARDCODED_LMAX > 5) {
        COMPUTE_SPH_DERIVATIVE_L6(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }
}

template <bool DO_DERIVATIVES, bool NORMALIZED, int HARDCODED_LMAX>
inline void hardcoded_sph_sample(const double *xyz_i, double *sph_i, double *dsph_i, int size_y) {
    auto x = xyz_i[0];
    auto y = xyz_i[1];
    auto z = xyz_i[2];
    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    double ir=0;

    if constexpr(NORMALIZED) {
        auto ir2 = 1.0/(x2+y2+z2);
        ir = sqrt(ir2);
        x*=ir; y*=ir; z*=ir;
        x2*=ir2; y2*=ir2; z2*=ir2;
    }
    
    hardcoded_sph_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i);

    if constexpr (DO_DERIVATIVES) {
        double *dxsph_i = dsph_i;
        double *dysph_i = dxsph_i + size_y;
        double *dzsph_i = dysph_i + size_y;
        hardcoded_sph_derivative_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
        if constexpr(NORMALIZED) {
            // corrects derivatives for normalization
            for (int k=0; k<size_y; ++k) {
                dxsph_i[k]*=ir;
                dysph_i[k]*=ir;
                dzsph_i[k]*=ir;
                auto tmp = (dxsph_i[k]*x+dysph_i[k]*y+dzsph_i[k]*z);
                dxsph_i[k] -= x*tmp;
                dysph_i[k] -= y*tmp;
                dzsph_i[k] -= z*tmp;
            }
        }
    }
}

template <bool DO_DERIVATIVES, bool NORMALIZED, int HARDCODED_LMAX>
void hardcoded_sph(int n_samples, const double *xyz, double *sph, double *dsph) {
    /*
        Cartesian Ylm calculator using the hardcoded expressions.
        Templated version, just calls _compute_sph_templated and
        _compute_dsph_templated functions within a loop.
    */
    #pragma omp parallel
    {
        const double *xyz_i = nullptr;
        double *sph_i = nullptr;
        double *dsph_i = nullptr;
        constexpr auto size_y = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);        

        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_i = xyz + i_sample * 3;
            sph_i = sph + i_sample * size_y;
            if constexpr (DO_DERIVATIVES) {
                dsph_i = dsph + i_sample * size_y * 3;
            }
            hardcoded_sph_sample<DO_DERIVATIVES, NORMALIZED, HARDCODED_LMAX>(xyz_i, sph_i, dsph_i, size_y);
        }
    }
}

template <bool DO_DERIVATIVES, int HARDCODED_LMAX>
static inline void generic_sph_l_channel(int l, 
    double x, double y, double z, double rxy, double twoz, 
    double *sph_i,
    double *dxsph_i, double *dysph_i, double *dzsph_i,
    const double *pk,
    const double *qlmk,
    std::vector<double> &c,
    std::vector<double> &s
)
{    
    // working space for the recursive evaluation of Qlm and Q(l-1)m
    double qlm_2, qlm_1, qlm_0;
    double ql1m_2, ql1m_1, ql1m_0;
    qlm_2 = qlmk[l]; // fetches the pre-computed Qll    

    // l=+-m
    auto pq = qlm_2 * pk[l];
    auto pdq = 0.0;
    auto pdqx = 0.0;
    auto pdqy = 0.0;

    sph_i[-l] = pq * s[l];
    sph_i[+l] = pq * c[l];    

    if constexpr (DO_DERIVATIVES) {
        pq *= l;
        dxsph_i[-l] = pq * s[l - 1];
        dysph_i[-l] = dxsph_i[l] = pq * c[l - 1];
        dysph_i[l] = -dxsph_i[-l]; 
        dzsph_i[-l] = 0;
        dzsph_i[l] = 0;
        ql1m_2 = 0;
    }

    // l=+-(m-1)
    qlm_1 = -z*qlm_2;
    pq = qlm_1 * pk[l - 1];
    sph_i[-l + 1] = pq * s[l - 1];
    sph_i[+l - 1] = pq * c[l - 1];

    if constexpr (DO_DERIVATIVES) {
        pq *= (l - 1);
        dxsph_i[-l + 1] = pq * s[l - 2];
        dysph_i[-l + 1] = dxsph_i[l - 1] = pq * c[l - 2];         
        dysph_i[l - 1] = -dxsph_i[-l + 1]; 

        // uses Q(l-1)(l-1) to initialize the other recursion
        ql1m_1 = qlmk[-1];
        pdq = pk[l - 1] * (l + l - 1) * ql1m_1;
        dzsph_i[-l + 1] = pdq * s[l - 1];
        dzsph_i[l - 1] = pdq * c[l - 1];
    }

    // and now do the other m's, decrementally
    auto twomz = l * twoz; // compute decrementally to hold 2(m+1)z
    for (auto m = l - 2; m > HARDCODED_LMAX - 1; --m) {
        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1; qlm_1 = qlm_0; // shift
        
        pq = qlm_0 * pk[m];
        sph_i[-m] = pq * s[m];
        sph_i[+m] = pq * c[m];

        if constexpr (DO_DERIVATIVES) {
            pq *= m;
            ql1m_0 = qlmk[m-l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1; ql1m_1 = ql1m_0; // shift
            
            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dxsph_i[-m] = (pdqx * s[m] + pq * s[m - 1]);
            dxsph_i[+m] = (pdqx * c[m] + pq * c[m - 1]);
            pdqy = pdq * y;
            dysph_i[-m] = (pdqy * s[m] + pq * c[m - 1]);
            dysph_i[m] = (pdqy * c[m] - pq * s[m - 1]);
            pdq = pk[m] * (l + m) * ql1m_1;
            dzsph_i[-m] = pdq * s[m];
            dzsph_i[m] = pdq * c[m];
        }
    }
    for (auto m = HARDCODED_LMAX - 1; m > 0; --m) {
        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1; qlm_1 = qlm_0; // shift
        
        pq = qlm_0 * pk[m];
        sph_i[-m] = pq * s[m];
        sph_i[+m] = pq * c[m];

        if constexpr (DO_DERIVATIVES) {
            pq *= m;
            ql1m_0 = qlmk[m-l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1; ql1m_1 = ql1m_0; // shift
            
            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dxsph_i[-m] = (pdqx * s[m] + pq * s[m - 1]);
            dxsph_i[+m] = (pdqx * c[m] + pq * c[m - 1]);
            pdqy = pdq * y;
            dysph_i[-m] = (pdqy * s[m] + pq * c[m - 1]);
            dysph_i[m] = (pdqy * c[m] - pq * s[m - 1]);
            pdq = pk[m] * (l + m) * ql1m_1;
            dzsph_i[-m] = pdq * s[m];
            dzsph_i[m] = pdq * c[m];
        }
    }

    // m=0
    qlm_0 = qlmk[0] * (twoz * qlm_1 + rxy * qlm_2);  
    sph_i[0] = qlm_0 * pk[0];

    if constexpr (DO_DERIVATIVES) {
        ql1m_0 = qlmk[-l] * (twoz * ql1m_1 + rxy * ql1m_2);
        ql1m_2 = ql1m_1; ql1m_1 = ql1m_0; // shift
        // derivatives
        dxsph_i[0] = pk[0] * x *ql1m_2; 
        dysph_i[0] = pk[0] * y *ql1m_2; 
        dzsph_i[0] = pk[0] * l *ql1m_1; 
    }
}

template <bool DO_DERIVATIVES, bool NORMALIZED, int HARDCODED_LMAX>
static inline void generic_sph_sample(int l_max,
    const double *prefactors,
    const double *xyz_i,
    double *sph_i,
    double *dsph_i,
    int size_y,
    int size_q,
    std::vector<double> &c,
    std::vector<double> &s
) {

    double ir = 0.0;
    double* dxsph_i = nullptr;
    double* dysph_i = nullptr;
    double* dzsph_i = nullptr;     
    // gets an index to some factors that enter the Qlm iteration,
    // and are pre-computed together with the prefactors
    const double *qlmfactor = prefactors+size_q;

    /* k is a utility index to traverse lm arrays. we store sph in
    a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
    so we often write a nested loop on l and m and track where we
    got by incrementing a separate index k. */
    int k = 0;
    
    auto x = xyz_i[0];
    auto y = xyz_i[1];
    auto z = xyz_i[2];
    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    if constexpr(NORMALIZED) {
        auto ir2 = 1.0/(x2+y2+z2);
        ir = sqrt(ir2);
        x*=ir; y*=ir; z*=ir;
        x2*=ir2; y2*=ir2; z2*=ir2;
    }
    auto twoz = 2 * z;
    auto rxy = x2 + y2;
    
    // these are the hard-coded, low-lmax sph
    hardcoded_sph_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i);

    if constexpr (DO_DERIVATIVES) {
        // updates the pointer to the derivative storage
        dxsph_i = dsph_i;
        dysph_i = dxsph_i + size_y;
        dzsph_i = dysph_i + size_y;

        // these are the hard-coded, low-lmax sph
        hardcoded_sph_derivative_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
    }

    /* These are scaled version of cos(m phi) and sin(m phi).
        Basically, these are cos and sin multiplied by r_xy^m,
        so that they are just plain polynomials of x,y,z.    */
    // help the compiler unroll the first part of the loop
    int m = 0;
    for (m = 1; m < HARDCODED_LMAX + 1; ++m) {
        c[m] = c[m - 1] * x - s[m - 1] * y;
        s[m] = c[m - 1] * y + s[m - 1] * x;
    }
    for (; m < l_max + 1; m++) {
        c[m] = c[m - 1] * x - s[m - 1] * y;
        s[m] = c[m - 1] * y + s[m - 1] * x;
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
        dxsph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dysph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
        dzsph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
    }

    auto pk = prefactors+k;
    auto qlmk = qlmfactor+k;
    for (int l = HARDCODED_LMAX + 1; l < l_max + 1; l++) {
        generic_sph_l_channel<DO_DERIVATIVES, HARDCODED_LMAX>(l, x, y, z, rxy, twoz, 
                    sph_i, dxsph_i, dysph_i, dzsph_i, pk, qlmk, c, s);

        // shift pointers & indexes to the next l block
        qlmk += l+1; pk+=l+1;
        sph_i += 2 * l + 2;

        if constexpr(DO_DERIVATIVES) {
            dxsph_i += 2 * l + 2;
            dysph_i += 2 * l + 2;
            dzsph_i += 2 * l + 2;
        }
    }

    if constexpr(DO_DERIVATIVES && NORMALIZED) {
        // corrects derivatives for normalization
        dxsph_i = dsph_i;
        dysph_i = dxsph_i + size_y;
        dzsph_i = dysph_i + size_y;

        for (k=0; k<size_y; ++k) {
            auto tmp = (dxsph_i[k]*x+dysph_i[k]*y+dzsph_i[k]*z);
            dxsph_i[k] = (dxsph_i[k]-x*tmp)*ir;
            dysph_i[k] = (dysph_i[k]-y*tmp)*ir;
            dzsph_i[k] = (dzsph_i[k]-z*tmp)*ir;
        }
    }
}


template <bool DO_DERIVATIVES, bool NORMALIZED, int HARDCODED_LMAX>
void generic_sph(
    int n_samples,
    int l_max,
    const double *prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    /*
        Implementation of the general case, but start at HARDCODED_LMAX and use
        hard-coding before that.

        Some general implementation remarks:
        - we use an alternative iteration for the Qlm that avoids computing the
          low-l section
        - we compute at the same time Qlm and the corresponding Ylm, to reuse
          more of the pieces and stay local in memory. we use `if constexpr`
          to avoid runtime branching in the DO_DERIVATIVES=true/false cases
        - we explicitly split the loops in a fixed-length (depending on the
          template parameter HARDCODED_LMAX) and a variable lenght one, so that
          the compiler can choose to unroll if it makes sense
        - there's a bit of pointer gymnastics because of the contiguous storage
          of the irregularly-shaped Q[l,m] and Y[l,m] arrays
    */

    // implementation assumes to use hardcoded expressions for at least l=0,1
    static_assert(HARDCODED_LMAX>=1, "Cannot call the generic Ylm calculator for l<=1.");

    const auto size_y = (l_max + 1) * (l_max + 1);
    const auto size_q = (l_max + 1) * (l_max + 2) / 2;
    
    #pragma omp parallel
    {
        // thread-local storage arrays for Qlm (modified associated Legendre
        // polynomials) and terms corresponding to (scaled) cosine and sine of
        // the azimuth
        auto c = std::vector<double>(l_max + 1, 0.0);
        auto s = std::vector<double>(l_max + 1, 0.0);
        
        // pointers to the sections of the output arrays that hold Ylm and derivatives
        // for a given point
        double* sph_i = nullptr;
        double* dsph_i = nullptr;   
        
        // also initialize the sine and cosine, these never change
        c[0] = 1.0;
        s[0] = 0.0;

        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            auto xyz_i = xyz+i_sample*3; 
            // pointer to the segment that should store the i_sample sph
            sph_i = sph + i_sample * size_y;
            if constexpr (DO_DERIVATIVES) {
                // updates the pointer to the derivative storage
                dsph_i = dsph + i_sample * 3 * size_y;
            }            
            
            generic_sph_sample<DO_DERIVATIVES, NORMALIZED, HARDCODED_LMAX>(l_max, prefactors, xyz_i, sph_i, dsph_i,
                                size_y, size_q, c, s);
        }
    }
}

#endif
