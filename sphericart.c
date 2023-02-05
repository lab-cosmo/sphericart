#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if defined(__INTEL_COMPILER)
    #include "mkl.h"
#else
    #include "cblas.h"
#endif

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


void cartesian_spherical_harmonics_naive(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {

    double* r_sq = (double*) malloc(sizeof(double)*n_samples);
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        r_sq[i_sample] = 0.0;
        for (int alpha = 0; alpha < 3; alpha++) {
            r_sq[i_sample] += xyz[i_sample*3+alpha]*xyz[i_sample*3+alpha];
        }
    }

    double* q = (double*) malloc(sizeof(double)*n_samples*(l_max+1)*(l_max+2)/2);
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        q[i_sample*(l_max+1)*(l_max+2)/2+0+0] = 1.0;
        for (int m = 1; m < l_max+1; m++) {
            q[i_sample*(l_max+1)*(l_max+2)/2+m*(m+1)/2+m] = -(2*m+1)*q[i_sample*(l_max+1)*(l_max+2)/2+(m-1)*m/2+(m-1)];
            q[i_sample*(l_max+1)*(l_max+2)/2+m*(m+1)/2+(m-1)] = (2*m-1)*xyz[i_sample*3+2]*q[i_sample*(l_max+1)*(l_max+2)/2+(m-1)*m/2+(m-1)];
        }
        for (int m = 0; m < l_max-1; m++) {
            for (int l = m+2; l < l_max+1; l++) {
                q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+m] = ((2*l-1)*xyz[i_sample*3+2]*q[i_sample*(l_max+1)*(l_max+2)/2+(l-1)*l/2+m]-(l+m-1)*q[i_sample*(l_max+1)*(l_max+2)/2+(l-2)*(l-1)/2+m]*r_sq[i_sample])/(l-m);
            }
        }
    }

    double *c = (double*) malloc(sizeof(double)*n_samples*(l_max+1));
    double *s = (double*) malloc(sizeof(double)*n_samples*(l_max+1));

    for (int i_sample = 0; i_sample < n_samples; i_sample++) {
        c[i_sample*(l_max+1)] = 1.0;
        s[i_sample*(l_max+1)] = 0.0;
        for (int m = 1; m < l_max+1; m++) {
            c[i_sample*(l_max+1)+m] = c[i_sample*(l_max+1)+(m-1)]*xyz[i_sample*3+0]-s[i_sample*(l_max+1)+(m-1)]*xyz[i_sample*3+1];
            s[i_sample*(l_max+1)+m] = c[i_sample*(l_max+1)+(m-1)]*xyz[i_sample*3+1]+s[i_sample*(l_max+1)+(m-1)]*xyz[i_sample*3+0];
        }
    }

    sph[0] = q[0] + c[0] + s[0];

    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        for (int l=0; l<l_max+1; l++) {
            for (int m=-l; m<0; m++) {
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+(-m)]*s[i_sample*(l_max+1)+(-m)];
            }
            sph[i_sample*(l+1)*(l+1)+l*l+l+0] = prefactors[l*l+l+0]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+0]*sqrt(2.0);
            for (int m=1; m<l_max+1; m++) {
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+m]*c[i_sample*(l_max+1)+m];
            }
        }
    }
    
    if (dsph != NULL) {
        // computes derivatives
    }

    free(q);
    free(c);
    free(s);
}


void cartesian_spherical_harmonics_cache(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {

    double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
    double* c = (double*) malloc(sizeof(double)*(l_max+1));
    double* s = (double*) malloc(sizeof(double)*(l_max+1));
    
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        double x = xyz[i_sample*3+0];
        double y = xyz[i_sample*3+1];
        double z = xyz[i_sample*3+2];
        double r_sq = x*x+y*y+z*z;

        q[0+0] = 1.0;
        for (int m = 1; m < l_max+1; m++) {
            q[m*(m+1)/2+m] = -(2*m+1)*q[(m-1)*m/2+(m-1)];
            q[m*(m+1)/2+(m-1)] = (2*m-1)*z*q[(m-1)*m/2+(m-1)];
        }
        for (int m = 0; m < l_max-1; m++) {
            for (int l = m+2; l < l_max+1; l++) {
                q[l*(l+1)/2+m] = ((2*l-1)*z*q[(l-1)*l/2+m]-(l+m-1)*q[(l-2)*(l-1)/2+m]*r_sq)/(l-m);
            }
        }

        c[0] = 1.0;
        s[0] = 0.0;
        for (int m = 1; m < l_max+1; m++) {
            c[m] = c[m-1]*x-s[m-1]*y;
            s[m] = c[m-1]*y+s[m-1]*x;
        }

        for (int l=0; l<l_max+1; l++) {
            for (int m=-l; m<0; m++) {
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[l*(l+1)/2+(-m)]*s[-m];
            }
            sph[i_sample*(l+1)*(l+1)+l*l+l+0] = prefactors[l*l+l+0]*q[l*(l+1)/2+0]*sqrt(2.0);
            for (int m=1; m<l_max+1; m++) {
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[l*(l+1)/2+m]*c[m];
            }
        }

        if (dsph != NULL) {
            // computes derivatives
        }

    }

    free(q);
    free(c);
    free(s);
}


void cartesian_spherical_harmonics_parallel(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {

    #pragma omp parallel
    {

        double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        double* c = (double*) malloc(sizeof(double)*(l_max+1));
        double* s = (double*) malloc(sizeof(double)*(l_max+1));
        
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            double x = xyz[i_sample*3+0];
            double y = xyz[i_sample*3+1];
            double z = xyz[i_sample*3+2];
            double r_sq = x*x+y*y+z*z;

            q[0+0] = 1.0;
            for (int m = 1; m < l_max+1; m++) {
                q[m*(m+1)/2+m] = -(2*m+1)*q[(m-1)*m/2+(m-1)];
                q[m*(m+1)/2+(m-1)] = (2*m-1)*z*q[(m-1)*m/2+(m-1)];
            }
            for (int m = 0; m < l_max-1; m++) {
                for (int l = m+2; l < l_max+1; l++) {
                    q[l*(l+1)/2+m] = ((2*l-1)*z*q[(l-1)*l/2+m]-(l+m-1)*q[(l-2)*(l-1)/2+m]*r_sq)/(l-m);
                }
            }

            c[0] = 1.0;
            s[0] = 0.0;
            for (int m = 1; m < l_max+1; m++) {
                c[m] = c[m-1]*x-s[m-1]*y;
                s[m] = c[m-1]*y+s[m-1]*x;
            }

            for (int l=0; l<l_max+1; l++) {
                for (int m=-l; m<0; m++) {
                    sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[l*(l+1)/2+(-m)]*s[-m];
                }
                sph[i_sample*(l+1)*(l+1)+l*l+l+0] = prefactors[l*l+l+0]*q[l*(l+1)/2+0]*sqrt(2.0);
                for (int m=1; m<l_max+1; m++) {
                    sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[l*l+l+m]*q[l*(l+1)/2+m]*c[m];
                }
            }

            if (dsph != NULL) {
                // computes derivatives
            }

        }

        free(q);
        free(c);
        free(s);
    }

}
