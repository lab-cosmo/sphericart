void cartesian_spherical_harmonics_naive(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_cache(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);
void compute_sph_prefactors(unsigned int l_max, double *factors);
