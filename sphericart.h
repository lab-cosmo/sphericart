void compute_sph_prefactors(unsigned int l_max, double *factors);
void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);

