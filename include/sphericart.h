#ifndef SPHERICART_H
#define SPHERICART_H

#include "sphericart/exports.h"

#ifdef __cplusplus
extern "C" {
#endif


void SPHERICART_EXPORT sphericart_compute_sph_prefactors(int l_max, double *factors);

void SPHERICART_EXPORT sphericart_cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);


#ifdef __cplusplus
}
#endif

#endif
