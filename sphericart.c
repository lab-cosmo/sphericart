#include "math.h"

#define LM_IDX(l, m) l*l+l+m

void compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
    */

    unsigned int k=0;
    for (unsigned int l=0; l<=l_max; ++l) {
        double factor = (2*l+1) 
        for (int m=-l; m<=l; ++m) {
            factors[k] = 
            ++k; 
        }
        
    }
}


void cartesian_spherical_harmonics(unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {
    /* 
        Computes the spherical harmonics
    */


    //...
    if (dsph != NULL) {
        // computes derivatives

    }    
}