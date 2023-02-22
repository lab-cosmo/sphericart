#ifndef SPHERICART_TEMPLATES_HPP
#define SPHERICART_TEMPLATES_HPP

#include <cstring>

#include "macros.hpp"

template <int HARDCODED_LMAX>
inline void hardcoded_sph_template(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
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
void hardcoded_sph(unsigned int n_samples, double *xyz, double *sph, double *dsph) {
    #pragma omp parallel
    {
        double x, y, z, x2=0, y2=0, z2=0;
        double *xyz_i, *sph_i;
        constexpr int size_y=((HARDCODED_LMAX+1) * (HARDCODED_LMAX+1));
        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {
            xyz_i = xyz + i_sample * 3;
            x = xyz_i[0];
            y = xyz_i[1];
            z = xyz_i[2];
            if constexpr (HARDCODED_LMAX > 2) {
                x2 = x * x;
                y2 = y * y;
                z2 = z * z;
            }
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
    unsigned int n_samples,
    unsigned int l_max,
    const double *prefactors,
    double *xyz,
    double *sph,
    double *dsph
) {
    // general case, but start at HARDCODED_LMAX and use hard-coding before that
    #pragma omp parallel
    {
        // storage arrays for Qlm (modified associated Legendre polynomials)
        // and terms corresponding to (scaled) cosine and sine of the azimuth
        double *q = (double *)malloc(sizeof(double) * (l_max + 1) * (l_max + 2) / 2);
        double *c = (double *)malloc(sizeof(double) * (l_max + 1));
        double *s = (double *)malloc(sizeof(double) * (l_max + 1));

        // temporaries to store prefactor*q and dq
        double pq, pdq, pdqx, pdqy;
        int l, m, k, size_y = (l_max + 1) * (l_max + 1), size_q = (l_max + 1) * (l_max + 2) / 2;

        // precomputes some factors that enter the Qlm iteration.
        // TODO: Probably worth pre-computing together with the prefactors,
        // more for consistency than for efficiency
        double *qlmfactor = (double *)malloc(sizeof(double) * size_q);
        k = (HARDCODED_LMAX) * (HARDCODED_LMAX + 1) / 2;
        for (l = HARDCODED_LMAX; l < l_max + 1; ++l) {
            for (m = l - 2; m >= 0; --m) {
                qlmfactor[k + m] = -1.0 / ((l + m + 1) * (l - m));
            }
            k += l + 1;
        }

        // precompute the Qll's (that are constant)
        q[0 + 0] = 1.0;
        k = 1;
        for (l = 1; l < l_max + 1; l++) {
            q[k + l] = -(2 * l - 1) * q[k - 1];
            k += l + 1;
        }

        // also initialize the sine and cosine, these never change
        c[0] = 1.0;
        s[0] = 0.0;

        /* k is a utility index to traverse lm arrays. we store sph in
        a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
        so we often write a nested loop on l and m and track where we
        got by incrementing a separate index k. */
        #pragma omp for
        for (int i_sample = 0; i_sample < n_samples; i_sample++) {

            double x = xyz[i_sample * 3 + 0];
            double y = xyz[i_sample * 3 + 1];
            double z = xyz[i_sample * 3 + 2];
            double twoz = 2 * z, twomz;
            double x2 = x * x, y2 = y * y, z2 = z * z;
            double rxy = x2 + y2;

            // pointer to the segment that should store the i_sample sph
            double *sph_i = sph + i_sample * size_y;
            double *dsph_i, *dxsph_i, *dysph_i, *dzsph_i;

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
            k = (HARDCODED_LMAX) * (HARDCODED_LMAX + 1) / 2;
            q[k + HARDCODED_LMAX - 1] = -z * q[k + HARDCODED_LMAX];
            twomz = (HARDCODED_LMAX)*twoz; // compute decrementally to hold 2(m+1)z
            for (m = HARDCODED_LMAX - 2; m >= 0; --m) {
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
            for (l = HARDCODED_LMAX + 1; l < l_max + 1; ++l) {
                // l=+-m
                pq = q[k + l] * prefactors[k + l];
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
