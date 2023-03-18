#include "sphericart.hpp"
#include "sphericart.h"

extern "C" {    
    sphericart_spherical_harmonics *sphericart_new(size_t l_max, char normalized) {
        return reinterpret_cast<sphericart_spherical_harmonics*>(new sphericart::SphericalHarmonics<double>(l_max, (bool) normalized)); 
    }
    sphericart_spherical_harmonics *sphericart_new_f(size_t l_max, char normalized) {
        return reinterpret_cast<sphericart_spherical_harmonics*>(new sphericart::SphericalHarmonics<float>(l_max, (bool) normalized)); 
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