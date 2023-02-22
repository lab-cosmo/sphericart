#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include "sphericart/exports.h"

namespace sphericart {

void SPHERICART_EXPORT compute_sph_prefactors(unsigned int l_max, double *factors);
void SPHERICART_EXPORT cartesian_spherical_harmonics(
    unsigned int n_samples,
    unsigned int l_max,
    const double* prefactors,
    double *xyz,
    double *sph,
    double *dsph
);

} // namespace sphericart

#endif
