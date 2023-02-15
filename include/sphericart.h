/**
 * This function calculates the spherical harmonics and, optionally, their derivatives 
 * for a set of 3D points.
 * 
 * @param l_max The maximum degree of spherical harmonics for which the prefactors will
 *      be calculated.
 * @param factors On entry, a (possibly uninitialized) array of size ``(l_max+1)*(l_max+2)/2``.
 *      On exit, it will contain the prefactors for the calculation of the spherical harmonics
 *      up to degree ``l_max``, in the order (l, m) = (0, 0), (1, 0), (1, 1), (2, 0),
 *      (2, 1), (2, 2), etc.
 */
void compute_sph_prefactors(unsigned int l_max, double *factors);

/**
 * This function calculates the spherical harmonics and, optionally, their derivatives 
 * for a set of 3D points.
 * 
 * @param n_samples The number of 3D points for which the spherical harmonics will be calculated.
 * @param l_max The maximum degree of the spherical harmonics to be calculated.
 * @param prefactors Prefactors for the spherical harmonics as computed by the ``compute_sph_prefactors`` function.
 * @param xyz An array of size ``(n_samples)*3``. It contains the Cartesian coordinates of the 3D points for 
 *      which the spherical harmonics are to be computed, organized along two dimensions. The outer dimension
 *      is ``n_samples`` long, accounting for different samples, while the inner dimension has size 3 and it
 *      represents the x, y, and z coordinates respectively.
 * @param sph On entry, a (possibly uninitialized) array of size ``n_samples*(l_max+1)*(l_max+1)``.
 *      On exit, this array will contain the spherical harmonics organized along two dimensions. The leading 
 *      dimension is ``n_samples`` long and it represents the different samples, while the inner dimension is 
 *      ``(l_max+1)*(l_max+1)`` long and it contains the spherical harmonics. These are laid out in lexicographic
 *      order. For example, if ``l_max=2``, it will contain (l, m) = (0, 0), (1, -1), (1, 0), (1, 1), 
 *      (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), in this order.
 * @param dsph On entry, either ``NULL`` or a (possibly uninitialized) array of size ``n_samples*3*(l_max+1)*(l_max+1)``.
 *      If ``dsph`` is ``NULL``, the spherical harmonics' derivatives will not be calculated. Otherwise,
 *      on exit, this array will contain the spherical harmonics' derivatives organized along three dimensions. 
 *      As for the ``sph`` parameter, the leading dimension represents the different samples, while the inner-most 
 *      dimension is ``(l_max+1)*(l_max+1)``, and it represents the degree and order of the spherical harmonics 
 *      (again, organized in lexicographic order). The intermediate dimension corresponds to different spatial derivatives
 *      of the spherical harmonics: x, y, and z, respectively.
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
