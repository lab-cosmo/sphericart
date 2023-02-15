/**
 * This function calculates the spherical harmonics and, optionally, their derivatives 
 * for a set of 3D points.
 * 
 * @param l_max The number of 3D points for which the spherical harmonics will be calculated.
 * @param factors ...
 */
void compute_sph_prefactors(unsigned int l_max, double *factors);

/**
 * This function calculates the spherical harmonics and, optionally, their derivatives 
 * for a set of 3D points.
 * 
 * @param n_samples The number of 3D points for which the spherical harmonics will be calculated.
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the ``compute_sph_prefactors`` function.
 * @param xyz ...
 * @param sph ...
 * @param dsph ...
 */
void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);

// Undocumented functions (for benchmarking and testing purposes)
void cartesian_spherical_harmonics_l0(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l1(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l2(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l3(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l4(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_l5(unsigned int n_samples, double *xyz, double *sph, double *dsph);
void cartesian_spherical_harmonics_generic(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph);
