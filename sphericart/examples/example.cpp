/** @file example.cpp
 *  @brief Usage example for the C++ API
*/

#include "sphericart.hpp"
#include <cmath>
#include <cstdio>

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {    
    // hard-coded parameters for the example
    size_t n_samples = 10000;
    size_t l_max = 10;

    // initializes samples
    auto xyz = std::vector<double>(n_samples*3, 0.0);
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz[i] = (double)rand()/ (double) RAND_MAX *2.0-1.0;
    }

    // to avoid unnecessary allocations, the class assumes pre-allocated arrays
    auto sph = std::vector<double>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph = std::vector<double>(n_samples*3*(l_max+1)*(l_max+1), 0.0);

    // the class declaration initializes buffers and numerical factors
    auto my_sph = sphericart::SphericalHarmonics<double>(l_max, false);
    
    // once initialized, the class can be called on arrays of points or on 
    // individual samples - this is deduced from the size of the array
    my_sph.compute(xyz, sph, dsph);
    auto xyz_sample = std::vector<double>(3, 0.0);
    auto sph_sample = std::vector<double>((l_max+1)*(l_max+1), 0.0);
    auto dsph_sample = std::vector<double>(3*(l_max+1)*(l_max+1), 0.0);
    my_sph.compute(xyz_sample, sph_sample, dsph_sample); 

    // the class is templated, so one can also use 16-bit float operations
    auto xyz_f = std::vector<float>(n_samples*3, 0.0);
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz_f[i] = (float) xyz[i];
    }
    auto sph_f = std::vector<float>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph_f = std::vector<float>(n_samples*3*(l_max+1)*(l_max+1), 0.0);
    
    auto my_sph_f = sphericart::SphericalHarmonics<float>(l_max, false);
    my_sph_f.compute(xyz_f, sph_f, dsph_f); 

    double sph_error = 0.0, sph_norm = 0.0;
    for (size_t i=0; i<n_samples*(l_max+1)*(l_max+1); ++i) {
        sph_error += (sph_f[i] - sph[i])*(sph_f[i] - sph[i]);
        sph_norm +=  sph[i]*sph[i];
    }
    printf("Float vs double relative error: %12.8e\n", sqrt(sph_error/sph_norm));
}