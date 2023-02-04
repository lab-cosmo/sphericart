#include "sphericart.h"
#include "stdlib.h"
#include "stdio.h"

/* main.c */
int main(int argc, char *argv[]) {
    unsigned int n_samples = 10000;
    unsigned int l_max = 12;
    double *prefactors = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+1));

    compute_sph_prefactors(l_max, prefactors);

    double *xyz = (double*) malloc(sizeof(double)*n_samples*3);
    double *sph = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+1));
    
    cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, NULL);    
}