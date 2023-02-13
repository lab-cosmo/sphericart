void compute_sph_prefactors(unsigned int l_max, double *factors);
void cartesian_spherical_harmonics_l0(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l1(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l2(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l3(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_hybrid(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);

