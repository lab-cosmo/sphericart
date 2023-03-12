#include "sphericart.hpp"
#include "sphericart.h"


extern "C" void sphericart_compute_sph_prefactors(int l_max, sphericart_yq_buffer *factors) {
    sphericart::compute_sph_prefactors(l_max, (sphericart::yq_buffer<double>*) factors);
}

extern "C" void sphericart_cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const sphericart_yq_buffer* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    sphericart::cartesian_spherical_harmonics(n_samples, l_max, (const sphericart::yq_buffer<double>*) prefactors, xyz, sph, dsph);
}

extern "C" void sphericart_normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const sphericart_yq_buffer* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    sphericart::normalized_spherical_harmonics(n_samples, l_max, (const sphericart::yq_buffer<double>*) prefactors, xyz, sph, dsph);
}
