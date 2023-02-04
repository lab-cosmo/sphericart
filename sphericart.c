#include "math.h"
#include "stdio.h"

#define LM_IDX(l, m) l*l+l+m

void compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        sqrt((2l+1)/pi (l-|m|)!/(l+m)!)
    */

    unsigned int k=0; // quick access index
    for (unsigned int l=0; l<=l_max; ++l) {
        double factor = (2*l+1)/(2*M_PI);
        k+=l;
        factors[k] = sqrt(factor);        
        for (int m=1; m<=l; ++m) {
            factor *= 1.0/(l*(l+1)+m*(1-m));
            factors[k-m] = factors[k+m] = sqrt(factor);         
        }
        k += l+1;
        printf("%d %d %e\n", l, k, factor);
    }
}


void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {
    /* 
        Computes the spherical harmonics
    */


    //...
    if (dsph != NULL) {
        // computes derivatives

    }    
}