/** @file example.c
 *  @brief Usage example for the C API
*/

#include "sphericart.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

int main(int argc, char *argv[]) {
    // hard-coded parameters for the example
    size_t n_samples = 10000;
    size_t l_max = 10;

    // initializes samples
    double *xyz = malloc(n_samples * 3 * sizeof(double));
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz[i] = (double)rand()/ (double) RAND_MAX *2.0-1.0;
    }

    // to avoid unnecessary allocations, the class assumes pre-allocated arrays
    double *sph = malloc(n_samples * (l_max + 1) * (l_max + 1) * sizeof(double));
    double *dsph = malloc(n_samples * 3 * (l_max + 1) * (l_max + 1) * sizeof(double));

    // the class declaration initializes buffers and numerical factors
    sphericart_calculator_t* calculator = sphericart_new(l_max, 0);

    sphericart_compute_array(calculator, n_samples, xyz, sph, NULL); // without derivatives
    sphericart_compute_array(calculator, n_samples, xyz, sph, dsph); // with derivatives

    // sample calculation - we reuse the same arrays, but only the first item is computed
    sphericart_compute_sample(calculator, xyz, sph, NULL);
    sphericart_compute_sample(calculator, xyz, sph, dsph);

    sphericart_calculator_f_t* calculator_f = sphericart_new_f(l_max, 0);

    // float versions
    float *xyz_f = malloc(n_samples * 3 * sizeof(float));
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz_f[i] = xyz[i];
    }
    float *sph_f = malloc(n_samples * (l_max + 1) * (l_max + 1) * sizeof(float));
    float *dsph_f = malloc(n_samples * 3 * (l_max + 1) * (l_max + 1) * sizeof(float));

    sphericart_compute_array_f(calculator_f, n_samples, xyz_f, sph_f, NULL); // without derivatives
    sphericart_compute_array_f(calculator_f, n_samples, xyz_f, sph_f, dsph_f); // with derivatives

    double sph_error = 0.0, sph_norm = 0.0;
    for (size_t i=0; i<n_samples*(l_max+1)*(l_max+1); ++i) {
        sph_error += (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
        sph_norm += sph[i]*sph[i];
    }
    printf("Float vs double relative error: %12.8e\n", sqrt(sph_error/sph_norm));

    // frees up data arrays and sph object pointers
    sphericart_delete(calculator);
    free(xyz);
    free(sph);
    free(dsph);

    sphericart_delete_f(calculator_f);
    free(xyz_f);
    free(sph_f);
    free(dsph_f);
}
