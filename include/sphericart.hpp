#ifndef SPHERICART_HPP
#define SPHERICART_HPP

#include <cstddef>

#include "sphericart/exports.h"

namespace sphericart {

void SPHERICART_EXPORT compute_sph_prefactors(int l_max, double *factors);

void SPHERICART_EXPORT cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
);

} // namespace sphericart

#endif
