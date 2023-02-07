#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
        Use an iterative formula to avoid computing a ratio
        of factorials. 
    */

    unsigned int k=0; // quick access index
    for (unsigned int l=0; l<=l_max; ++l) {
        double factor = (2*l+1)/(2*M_PI);
        factors[k] = sqrt(factor);        
        for (int m=1; m<=l; ++m) {
            factor *= 1.0/(l*(l+1)+m*(1-m));
            if (m % 2 == 0) {
                factors[k+m] = sqrt(factor);    
            } else {
                factors[k+m] = - sqrt(factor); 
            }     
        }
        k += l+1;
    }
}

void cartesian_spherical_harmonics_l0(unsigned int n_samples, double *xyz, double *sph, double *dsph) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            sph[i_sample] = 0.282094791773878;
            dsph[i_sample*3] = dsph[i_sample*3+1] = dsph[i_sample*3+2] = 0.0;
        }
    }
}

void cartesian_spherical_harmonics_l1(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {
    #pragma omp parallel
    {
        double *xyz_i, *sph_i, *dsph_i;
        
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {
            xyz_i = xyz+i_sample*3;
            sph_i = sph+i_sample*4;
            dsph_i = dsph+i_sample*4*3;
            // l=0 m=0
            sph_i[0] = 0.282094791773878;
            dsph_i[0] = dsph_i[4] = dsph_i[8] = 0.0;
            ++sph_i; 

            // l=1 m=-1,0,1
            sph_i[0] = 0.48860251190292*xyz_i[1];
            sph_i[1] = 0.48860251190292*xyz_i[2];
            sph_i[2] = 0.48860251190292*xyz_i[0];
            
            dsph_i[1] = 0.0; dsph_i[2] = 0.0; dsph_i[3] = 0.48860251190292;  //d/dx 
            dsph_i[5] = 0.48860251190292; dsph_i[6] = 0.0; dsph_i[7] = 0.0;  //d/dy
            dsph_i[9] = 0.0; dsph_i[10] = 0.48860251190292; dsph_i[11] = 0.0;  //d/dz

        }
    }
}

void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, 
            const double* prefactors, double *xyz, double *sph, double *dsph) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and 
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the 
        r^l term can be easily incorporated into any radial function or 
        added a posteriori (with the corresponding derivative).
    */

    #pragma omp parallel
    {
        // storage arrays for Qlm (modified associated Legendre polynomials)
        // and terms corresponding to (scaled) cosine and sine of the azimuth
        double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        double* c = (double*) malloc(sizeof(double)*(l_max+1));
        double* s = (double*) malloc(sizeof(double)*(l_max+1));

        // temporaries to store prefactor*q and dq
        double pq, pdq, pdqx, pdqy; 
        int size_y = (l_max+1)*(l_max+1);

        /* k is a utility index to traverse lm arrays. we store sph in 
           a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
           so we often write a nested loop on l and m and track where we
           got by incrementing a separate index k. */
        int k; 
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {

            double x = xyz[i_sample*3+0];
            double y = xyz[i_sample*3+1];
            double z = xyz[i_sample*3+2];
            double r_sq = x*x+y*y+z*z;

            // pointer to the segment that should store the i_sample sph
            double *sph_i = sph+i_sample*size_y;
            
            /* compute recursively the "Cartesian" associated Legendre polynomials Qlm.
               Qlm is defined as r^l/r_xy^m P_lm, and is a polynomial of x,y,z.
               These are computed with a recursive expression.
              */
            
            // Initialize the recursion
            q[0+0] = 1.0;
            k=1;
            for (int m = 1; m < l_max+1; m++) {
                q[k+m] = -(2*m-1)*q[k-1];
                q[k+(m-1)] = -z*q[k+m]; // (2*m-1)*z*q[k-1];
                k += m+1; 
            }

            // base index to traverse the Qlm. the initial index for q[lm] starts at l=2
            k = 3; 
            for (int l=2; l < l_max+1; ++l) {
                double twolz = (2*l-1)*z;
                double lmrsq = (l-2)*r_sq;
                for (int m=0; m < l-1; ++m) {
                    lmrsq += r_sq; // this computes (l-1+m) r_sq
                    q[k] = (twolz*q[k-l]-lmrsq*q[k-(2*l-1)])/(l-m);
                    ++k; 
                }
                k += 2; // we must skip the 2 that are already precomputed
            }

            /* These are scaled version of cos(m phi) and sin(m phi).
               Basically, these are cos and sin multiplied by r_xy^m,
               so that they are just plain polynomials of x,y,z.
            */
            c[0] = 1.0;
            s[0] = 0.0;
            for (int m = 1; m < l_max+1; m++) {
                c[m] = c[m-1]*x-s[m-1]*y;
                s[m] = c[m-1]*y+s[m-1]*x;
            }

            /* fill the (Cartesian) sph by combining Qlm and 
              sine/cosine phi-dependent factors. we use pointer 
              arithmetics to make sure spk_i always points at the 
              beginning of the appropriate memory segment. */
            sph_i[0] = q[0]*prefactors[0]*M_SQRT1_2;  //l=0
            k = 1; ++sph_i;
            for (int l=1; l<l_max+1; l++) {            
                sph_i[l] = q[k]*prefactors[k]*M_SQRT1_2;
                for (int m=1; m<l+1; m++) {
                    pq = q[k+m]*prefactors[k+m];
                    sph_i[l-m] = pq*s[m];
                    sph_i[l+m] = pq*c[m];
                }             
                k += l+1;
                sph_i += 2*l+1;
            }

            if (dsph != NULL) {
                // if the pointer is set, we compute the derivatives
                // we use the chain rule and some nice recursions

                // updates the pointer to the derivative storage
                double *dsph_i = dsph+i_sample*3*size_y; 

                
                k=0;
                // special case: l=0
                dsph_i[0] = dsph_i[size_y] = dsph_i[size_y*2] = 0;
                ++k; ++dsph_i;

                // special case: l=1
                dsph_i[1] = dsph_i[size_y+1] = 0;
                dsph_i[size_y*2+1] = prefactors[k]*1*q[k-1]*M_SQRT1_2;
                ++k; 
                pq=prefactors[k]*q[k];
                dsph_i[0] = pq*s[0];
                dsph_i[2] = pq*c[0];
                dsph_i[size_y+0] = pq*c[0];
                dsph_i[size_y+2] = -pq*s[0];
                dsph_i[size_y*2+0] = 0;
                dsph_i[size_y*2+2] = 0;                
                ++k;                        
                dsph_i+=3;

                // general case - iteration
                for (int l=2; l<l_max+1; l++) {
                    dsph_i[l] = prefactors[k]*x*q[k-l+1]*M_SQRT1_2;
                    dsph_i[size_y+l] = prefactors[k]*y*q[k-l+1]*M_SQRT1_2;
                    dsph_i[size_y*2+l] = prefactors[k]*l*q[k-l]*M_SQRT1_2;
                    
                    ++k;
                    for (int m=1; m<l-1; m++) {
                        // also includes a factor of m so we get the phi-dependent derivatives
                        pq=prefactors[k]*q[k]*m;  
                        pdq=prefactors[k]*q[k-l+1];
                        pdqx = pdq*x;
                        dsph_i[l-m] = (pdqx*s[m]+pq*s[m-1]);
                        dsph_i[l+m] = (pdqx*c[m]+pq*c[m-1]);
                        pdqy = pdq*y;
                        dsph_i[size_y+l-m] = (pdqy*s[m]+pq*c[m-1]);
                        dsph_i[size_y+l+m] = (pdqy*c[m]-pq*s[m-1]);
                        pdq=prefactors[k]*(l+m)*q[k-l];
                        dsph_i[size_y*2+l-m] = pdq*s[m];
                        dsph_i[size_y*2+l+m] = pdq*c[m];
                        ++k;
                    }

                    // do separately special cases that have lots of zeros
                    // m = l-1
                    pq=prefactors[k]*q[k]*(l-1); 
                    dsph_i[l-l+1] = pq*s[l-2];
                    dsph_i[l+l-1] = pq*c[l-2];
                    dsph_i[size_y+l-l+1] = pq*c[l-2];
                    dsph_i[size_y+l+l-1] = -pq*s[l-2];
                    pdq=prefactors[k]*(l+l-1)*q[k-l]; 
                    dsph_i[size_y*2+l-l+1] = pdq*s[l-1];
                    dsph_i[size_y*2+l+l-1] = pdq*c[l-1];
                    ++k;
                    //m=l
                    pq=prefactors[k]*q[k]*l; 
                    dsph_i[l-l] = pq*s[l-1];
                    dsph_i[l+l] = pq*c[l-1];
                    dsph_i[size_y+l-l] = pq*c[l-1];
                    dsph_i[size_y+l+l] = -pq*s[l-1];
                    dsph_i[size_y*2+l-l] = 0;
                    dsph_i[size_y*2+l+l] = 0;
                    ++k;
                    //advances the pointer for the sph derivatives
                    dsph_i += 2*l+1;  
                }
            }     
        }
        free(q);
        free(c);
        free(s);
    }
}
