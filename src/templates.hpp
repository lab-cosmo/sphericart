#ifndef SPHERICART_TEMPLATES_HPP
#define SPHERICART_TEMPLATES_HPP

/*
    Template implemntation of Cartesian Ylm calculators. 

    The template functions use compile-time `if constexpr()` constructs to 
    implement calculators for spherical harmonics that can handle different
    type of calls, e.g. with or without derivative calculations, and with
    different numbers of terms computed with hard-coded expressions.
*/

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
        Combines the macro hard-coded dYlm/d(x,y,z) calculators to get all the terms up to HC_LMAX. 
        This templated version evaluates the ifs at compile time avoiding unnecessary in-loop
        branching. 
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

template <bool DO_DERIVATIVES, int HARDCODED_LMAX>
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
        constexpr auto size_y = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);

        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_i = xyz + i_sample * 3;
            auto x = xyz_i[0];
            auto y = xyz_i[1];
            auto z = xyz_i[2];
            auto x2 = x * x;
            auto y2 = y * y;
            auto z2 = z * z;

            sph_i = sph + i_sample * size_y;
            hardcoded_sph_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i);

            if constexpr (DO_DERIVATIVES) {
                double *dxsph_i = dsph + i_sample * size_y * 3;
                double *dysph_i = dxsph_i + size_y;
                double *dzsph_i = dysph_i + size_y;
                hardcoded_sph_derivative_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
            }
        }
    }
}

template <bool DO_DERIVATIVES, int HARDCODED_LMAX>
void generic_sph(
    int n_samples,
    int l_max,
    const double *prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    // general case, but start at HARDCODED_LMAX and use hard-coding before that.
    // implementation assumes to use hardcoded expressions for at least l=0,1
    static_assert(HARDCODED_LMAX>=1, "Cannot call the generic Ylm calculator for l<=1.");

    #pragma omp parallel
    {
        // thread-local storage arrays for Qlm (modified associated Legendre
        // polynomials) and terms corresponding to (scaled) cosine and sine of
        // the azimuth
        double *q = (double *)malloc(sizeof(double) * (l_max + 1) * (l_max + 2) / 2);
        double *c = (double *)malloc(sizeof(double) * (l_max + 1));
        double *s = (double *)malloc(sizeof(double) * (l_max + 1));

        auto size_y = (l_max + 1) * (l_max + 1);
        auto size_q = (l_max + 1) * (l_max + 2) / 2;

        /* k is a utility index to traverse lm arrays. we store sph in
        a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
        so we often write a nested loop on l and m and track where we
        got by incrementing a separate index k. */
        auto k = (HARDCODED_LMAX) * (HARDCODED_LMAX + 1) / 2;
        
        // precomputes some factors that enter the Qlm iteration.
        // TODO: Probably worth pre-computing together with the prefactors,
        // more for consistency than for efficiency
        double *qlmfactor = (double *)malloc(sizeof(double) * size_q);
        for (int l = HARDCODED_LMAX; l < l_max + 1; l++) {
            for (int m = l - 2; m >= 0; --m) {
                qlmfactor[k + m] = -1.0 / ((l + m + 1) * (l - m));
            }
            k += l + 1;
        }

        // precompute the Qll's (that are constant)
        q[0 + 0] = 1.0;
        k = 1;
        for (int l = 1; l < l_max + 1; l++) {
            q[k + l] = -(2 * l - 1) * q[k - 1];
            k += l + 1;
        }

        // also initialize the sine and cosine, these never change
        c[0] = 1.0;
        s[0] = 0.0;
        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {

            auto x = xyz[i_sample * 3 + 0];
            auto y = xyz[i_sample * 3 + 1];
            auto z = xyz[i_sample * 3 + 2];
            auto twoz = 2 * z;
            auto x2 = x * x;
            auto y2 = y * y;
            auto z2 = z * z;
            auto rxy = x2 + y2;

            // pointer to the segment that should store the i_sample sph
            double* sph_i = sph + i_sample * size_y;
            double* dsph_i = nullptr;
            double* dxsph_i = nullptr;
            double* dysph_i = nullptr;
            double* dzsph_i = nullptr;

            // these are the hard-coded, low-lmax sph
            hardcoded_sph_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i);

            if constexpr (DO_DERIVATIVES) {
                // updates the pointer to the derivative storage
                dsph_i = dsph + i_sample * 3 * size_y;
                dxsph_i = dsph_i;
                dysph_i = dxsph_i + size_y;
                dzsph_i = dysph_i + size_y;

                hardcoded_sph_derivative_template<HARDCODED_LMAX>(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i);
            }

            /* These are scaled version of cos(m phi) and sin(m phi).
               Basically, these are cos and sin multiplied by r_xy^m,
               so that they are just plain polynomials of x,y,z.
            */

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
               beginning of the appropriate memory segment.
            */

            // We need also Qlm for l=HARDCODED_LMAX because it's used in the derivatives
            auto k = (HARDCODED_LMAX) * (HARDCODED_LMAX + 1) / 2;
            q[k + HARDCODED_LMAX - 1] = -z * q[k + HARDCODED_LMAX];
            auto twomz = (HARDCODED_LMAX) * twoz; // compute decrementally to hold 2(m+1)z
            for (int m = HARDCODED_LMAX - 2; m >= 0; --m) {
                twomz -= twoz;
                q[k + m] = qlmfactor[k + m] * (twomz * q[k + m + 1] + rxy * q[k + m + 2]);
            }

            // main loop!
            // k points at Q[l,0]; sph_i at Y[l,0] (mid-way through each l chunk)
            k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;
            sph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);

            if constexpr (DO_DERIVATIVES) {
                dxsph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
                dysph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
                dzsph_i += (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1 + 1);
            }
            for (int l = HARDCODED_LMAX + 1; l < l_max + 1; l++) {
                // l=+-m
                auto pq = q[k + l] * prefactors[k + l];
                auto pdq = 0.0;
                auto pdqx = 0.0;
                auto pdqy = 0.0;


                sph_i[-l] = pq * s[l];
                sph_i[+l] = pq * c[l];

                if constexpr (DO_DERIVATIVES) {
                    pq *= l;
                    dxsph_i[-l] = pq * s[l - 1];
                    dxsph_i[l] = pq * c[l - 1];
                    dysph_i[-l] = pq * c[l - 1];
                    dysph_i[l] = -pq * s[l - 1];
                    dzsph_i[-l] = 0;
                    dzsph_i[l] = 0;
                }

                // l=+-(m-1)
                q[k + l - 1] = -z * q[k + l];
                pq = q[k + l - 1] * prefactors[k + l - 1];
                sph_i[-l + 1] = pq * s[l - 1];
                sph_i[+l - 1] = pq * c[l - 1];

                if constexpr (DO_DERIVATIVES) {
                    pq *= (l - 1);
                    dxsph_i[-l + 1] = pq * s[l - 2];
                    dxsph_i[l - 1] = pq * c[l - 2];
                    dysph_i[-l + 1] = pq * c[l - 2];
                    dysph_i[l - 1] = -pq * s[l - 2];
                    pdq = prefactors[k + l - 1] * (l + l - 1) * q[k + l - 1 - l];
                    dzsph_i[-l + 1] = pdq * s[l - 1];
                    dzsph_i[l - 1] = pdq * c[l - 1];
                }

                // and now do the other m's, decrementally
                twomz = l * twoz; // compute decrementally to hold 2(m+1)z
                for (m = l - 2; m > HARDCODED_LMAX - 1; --m) {
                    twomz -= twoz;
                    q[k + m] = qlmfactor[k + m] * (twomz * q[k + m + 1] + rxy * q[k + m + 2]);
                    pq = q[k + m] * prefactors[k + m];
                    sph_i[-m] = pq * s[m];
                    sph_i[+m] = pq * c[m];

                    if constexpr (DO_DERIVATIVES) {
                        pq *= m;
                        pdq = prefactors[k + m] * q[k + m - l + 1];
                        pdqx = pdq * x;
                        dxsph_i[-m] = (pdqx * s[m] + pq * s[m - 1]);
                        dxsph_i[+m] = (pdqx * c[m] + pq * c[m - 1]);
                        pdqy = pdq * y;
                        dysph_i[-m] = (pdqy * s[m] + pq * c[m - 1]);
                        dysph_i[m] = (pdqy * c[m] - pq * s[m - 1]);
                        pdq = prefactors[k + m] * (l + m) * q[k + m - l];
                        dzsph_i[-m] = pdq * s[m];
                        dzsph_i[m] = pdq * c[m];
                    }
                }
                for (m = HARDCODED_LMAX - 1; m > 0; --m) {
                    twomz -= twoz;
                    q[k + m] = qlmfactor[k + m] * (twomz * q[k + m + 1] + rxy * q[k + m + 2]);
                    pq = q[k + m] * prefactors[k + m];
                    sph_i[-m] = pq * s[m];
                    sph_i[+m] = pq * c[m];

                    if constexpr (DO_DERIVATIVES) {
                        pq *= m;
                        pdq = prefactors[k + m] * q[k + m - l + 1];
                        pdqx = pdq * x;
                        dxsph_i[-m] = (pdqx * s[m] + pq * s[m - 1]);
                        dxsph_i[+m] = (pdqx * c[m] + pq * c[m - 1]);
                        pdqy = pdq * y;
                        dysph_i[-m] = (pdqy * s[m] + pq * c[m - 1]);
                        dysph_i[m] = (pdqy * c[m] - pq * s[m - 1]);
                        pdq = prefactors[k + m] * (l + m) * q[k + m - l];
                        dzsph_i[-m] = pdq * s[m];
                        dzsph_i[m] = pdq * c[m];
                    }
                }
                // m=0
                q[k] = qlmfactor[k] * (twoz * q[k + 1] + rxy * q[k + 2]);
                sph_i[0] = q[k] * prefactors[k];

                if constexpr (DO_DERIVATIVES) {
                    // derivatives
                    dxsph_i[0] = prefactors[k] * x * q[k - l + 1];
                    dysph_i[0] = prefactors[k] * y * q[k - l + 1];
                    dzsph_i[0] = prefactors[k] * l * q[k - l];

                    dxsph_i += 2 * l + 2;
                    dysph_i += 2 * l + 2;
                    dzsph_i += 2 * l + 2;
                }

                // shift pointers & indexes to the next l block
                k += l + 1;
                sph_i += 2 * l + 2;
            }
        }

        free(qlmfactor);
        free(q);
        free(c);
        free(s);
    }
}

#endif
