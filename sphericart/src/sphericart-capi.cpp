#include "sphericart.hpp"
#include "sphericart.h"

extern "C" {    
    sphericart_spherical_harmonics *sphericart_new(size_t l_max, char normalize) {
        return reinterpret_cast<sphericart_spherical_harmonics*>(new sphericart::SphericalHarmonics<double>(l_max, (bool) normalize)); 
    }
    sphericart_spherical_harmonics *sphericart_new_f(size_t l_max, char normalize) {
        return reinterpret_cast<sphericart_spherical_harmonics*>(new sphericart::SphericalHarmonics<float>(l_max, (bool) normalize)); 
    }
    
    void sphericart_delete(sphericart_spherical_harmonics* spherical_harmonics) {
        delete reinterpret_cast<sphericart::SphericalHarmonics<double>*>(spherical_harmonics);
    }

    void sphericart_compute_array(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const double* xyz, double* sph, double* dsph) {        
        if (dsph == nullptr) {
            reinterpret_cast<sphericart::SphericalHarmonics<double>*>(spherical_harmonics)->compute_array((int) n_samples, xyz, sph);   
        } else {
            reinterpret_cast<sphericart::SphericalHarmonics<double>*>(spherical_harmonics)->compute_array((int) n_samples, xyz, sph, dsph);
        }
    }
    void sphericart_compute_sample(sphericart_spherical_harmonics* spherical_harmonics, const double* xyz, double* sph, double* dsph) {
        if (dsph == nullptr) {
            reinterpret_cast<sphericart::SphericalHarmonics<double>*>(spherical_harmonics)->compute_sample(xyz, sph);   
        } else {
            reinterpret_cast<sphericart::SphericalHarmonics<double>*>(spherical_harmonics)->compute_sample(xyz, sph, dsph);
        }
    }

    void sphericart_compute_array_f(sphericart_spherical_harmonics* spherical_harmonics, size_t n_samples, const float* xyz, float* sph, float* dsph) {
        if (dsph == nullptr) {
            reinterpret_cast<sphericart::SphericalHarmonics<float>*>(spherical_harmonics)->compute_array((int) n_samples, xyz, sph);   
        } else {
            reinterpret_cast<sphericart::SphericalHarmonics<float>*>(spherical_harmonics)->compute_array((int) n_samples, xyz, sph, dsph);
        }
    }
    void sphericart_compute_sample_f(sphericart_spherical_harmonics* spherical_harmonics, const float* xyz, float* sph, float* dsph) {
        if (dsph == nullptr) {
            reinterpret_cast<sphericart::SphericalHarmonics<float>*>(spherical_harmonics)->compute_sample(xyz, sph);   
        } else {
            reinterpret_cast<sphericart::SphericalHarmonics<float>*>(spherical_harmonics)->compute_sample(xyz, sph, dsph);
        }
    }
} // extern "C"


extern "C" void sphericart_compute_sph_prefactors(int l_max, double *factors) {
        sphericart::compute_sph_prefactors(l_max, factors);
    }

extern "C" void sphericart_cartesian_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    sphericart::cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, dsph);
}

extern "C" void sphericart_normalized_spherical_harmonics(
    int n_samples,
    int l_max,
    const double* prefactors,
    const double *xyz,
    double *sph,
    double *dsph
) {
    sphericart::normalized_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, dsph);
}
