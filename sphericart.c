#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
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
            q[i_sample*(l_max+1)*(l_max+2)/2+m*(m+1)/2+m] = -(2*m-1)*q[i_sample*(l_max+1)*(l_max+2)/2+(m-1)*m/2+(m-1)];
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
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[(l*l+l)/2+(-m)]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+(-m)]*s[i_sample*(l_max+1)+(-m)];
            }
            sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+0] = prefactors[(l*l+l)/2+0]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+0]*M_SQRT1_2;
            for (int m=1; m<l+1; m++) {
                sph[i_sample*(l_max+1)*(l_max+1)+l*l+l+m] = prefactors[(l*l+l)/2+m]*q[i_sample*(l_max+1)*(l_max+2)/2+l*(l+1)/2+m]*c[i_sample*(l_max+1)+m];
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

    double* dqdx;
    double* dcdx;
    double* dsdx;
    double* dqdy;
    double* dcdy;
    double* dsdy;
    double* dqdz;

    if (dsph != NULL) {
        dqdx = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dqdy = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dqdz = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dcdx = (double*) malloc(sizeof(double)*(l_max+1));
        dsdx = (double*) malloc(sizeof(double)*(l_max+1));
        dcdy = (double*) malloc(sizeof(double)*(l_max+1));
        dsdy = (double*) malloc(sizeof(double)*(l_max+1));
    } 

    int k; // utility index
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        double x = xyz[i_sample*3+0];
        double y = xyz[i_sample*3+1];
        double z = xyz[i_sample*3+2];
        double r_sq = x*x+y*y+z*z;
        // pointer to the segment that should store the i_sample sph
        double *sph_i = sph+i_sample*(l_max+1)*(l_max+1); 

        q[0+0] = 1.0;
        k=1;
        for (int m = 1; m < l_max+1; m++) {
            q[k+m] = -(2*m-1)*q[k-1];
            q[k+(m-1)] = -z*q[k+m]; // (2*m-1)*z*q[k-1];
            k += m+1; 
        }

        /* Compute Qlm */
        k = 3; // Base index to traverse the Qlm. Initial index for q[lm] starts at l=2
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

        // pre-multiplies the Qlm with the prefactors because there's no need to do it twice below   
        /*     
        for (k=0; k<(l_max+1)*(l_max+2)/2; ++k) {
            q[k]*=prefactors[k];
        }
        */
        // Commented. We need an intact q for the derivatives. See workaround below to avoid the double multiplication.

        c[0] = 1.0;
        s[0] = 0.0;
        for (int m = 1; m < l_max+1; m++) {
            c[m] = c[m-1]*x-s[m-1]*y;
            s[m] = c[m-1]*y+s[m-1]*x;
        }

        // We fill the (cartesian) sph by combining Qlm and sine/cosine phi-dependent factors
        k = 0;
        for (int l=0; l<l_max+1; l++) {
            double pq = q[k]*prefactors[k];
            sph_i[l] = pq*M_SQRT1_2;
            for (int m=1; m<l+1; m++) {
                pq = q[k+m]*prefactors[k+m];
                sph_i[l-m] = pq*s[m];
                sph_i[l+m] = pq*c[m];
            }             
            k += l+1;
            sph_i += 2*l+1;
        }

        if (dsph != NULL) {
            double *dsph_i = dsph+i_sample*3*(l_max+1)*(l_max+1); 

            // Derivatives of q
            dqdz[0] = dqdy[0] = dqdx[0] = 0.0;  // l = m = 0
            for (int l = 1; l < l_max+1; l++) {
                dqdx[l*(l+1)/2+l] = 0.0;
                dqdy[l*(l+1)/2+l] = 0.0;
                dqdz[l*(l+1)/2+l] = 0.0;
                dqdx[l*(l+1)/2+l-1] = 0.0;
                dqdy[l*(l+1)/2+l-1] = 0.0;
                for (int m = 0; m < l; m++) {
                    if (m != l-1) {
                        dqdx[l*(l+1)/2+m] = x*q[(l-1)*l/2+m+1];
                        dqdy[l*(l+1)/2+m] = y*q[(l-1)*l/2+m+1];
                    }
                    dqdz[l*(l+1)/2+m] = (l+m)*q[(l-1)*l/2+m];
                }
            }

            // Derivatives of c, s
            dsdx[0] = 0.0;
            dcdx[0] = 0.0;
            dsdy[0] = 0.0;
            dcdy[0] = 0.0;
            for (int m = 1; m < l_max+1; m++) {
                dsdx[m] = s[m-1];
                dcdx[m] = c[m-1];
                dsdy[m] = c[m-1];
                dcdy[m] = -s[m-1];
            }

            // Chain rule:
            for (int l=0; l<l_max+1; l++) {
                dsph_i[(l_max+1)*(l_max+1)*0+l*l+l] = prefactors[l*(l+1)/2]*dqdx[l*(l+1)/2];
                dsph_i[(l_max+1)*(l_max+1)*1+l*l+l] = prefactors[l*(l+1)/2]*dqdy[l*(l+1)/2];
                dsph_i[(l_max+1)*(l_max+1)*2+l*l+l] = prefactors[l*(l+1)/2]*dqdz[l*(l+1)/2];
                for (int m=1; m<l+1; m++) {
                    dsph_i[(l_max+1)*(l_max+1)*0+l*l+l-m] = prefactors[l*(l+1)/2+m]*(dqdx[l*(l+1)/2+m]*s[m]+q[l*(l+1)/2+m]*dsdx[m]);
                    dsph_i[(l_max+1)*(l_max+1)*1+l*l+l-m] = prefactors[l*(l+1)/2+m]*(dqdy[l*(l+1)/2+m]*s[m]+q[l*(l+1)/2+m]*dsdy[m]);
                    dsph_i[(l_max+1)*(l_max+1)*2+l*l+l-m] = prefactors[l*(l+1)/2+m]*dqdz[l*(l+1)/2+m]*s[m];
                    dsph_i[(l_max+1)*(l_max+1)*0+l*l+l+m] = prefactors[l*(l+1)/2+m]*(dqdx[l*(l+1)/2+m]*c[m]+q[l*(l+1)/2+m]*dcdx[m]);
                    dsph_i[(l_max+1)*(l_max+1)*1+l*l+l+m] = prefactors[l*(l+1)/2+m]*(dqdy[l*(l+1)/2+m]*c[m]+q[l*(l+1)/2+m]*dcdy[m]);
                    dsph_i[(l_max+1)*(l_max+1)*2+l*l+l+m] = prefactors[l*(l+1)/2+m]*dqdz[l*(l+1)/2+m]*c[m];
                }
            }
        }     
    }

    if (dsph != NULL) {
        free(dqdx);
        free(dcdx);
        free(dsdx);
        free(dqdy);
        free(dcdy);
        free(dsdy);
        free(dqdz);
    }

    free(q);
    free(c);
    free(s);
}


void cartesian_spherical_harmonics_fast(unsigned int n_samples, unsigned int l_max, const double* prefactors, double *xyz, double *sph, double *dsph) {
    
    double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
    double* c = (double*) malloc(sizeof(double)*(l_max+1));
    double* s = (double*) malloc(sizeof(double)*(l_max+1));

    double* dqdx;
    double* dcdx;
    double* dsdx;
    double* dqdy;
    double* dcdy;
    double* dsdy;
    double* dqdz;

    double *dq;
    if (dsph != NULL) {
        dq = (double*) malloc(sizeof(double)*3*(l_max+1)*(l_max+2)/2);
        dqdx = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dqdy = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dqdz = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        dcdx = (double*) malloc(sizeof(double)*(l_max+1));
        dsdx = (double*) malloc(sizeof(double)*(l_max+1));
        dcdy = (double*) malloc(sizeof(double)*(l_max+1));
        dsdy = (double*) malloc(sizeof(double)*(l_max+1));
    } 

    int k; // utility index
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        double x = xyz[i_sample*3+0];
        double y = xyz[i_sample*3+1];
        double z = xyz[i_sample*3+2];
        double r_sq = x*x+y*y+z*z;
        // pointer to the segment that should store the i_sample sph
        double *sph_i = sph+i_sample*(l_max+1)*(l_max+1); 

        q[0+0] = 1.0;
        k=1;
        for (int m = 1; m < l_max+1; m++) {
            q[k+m] = -(2*m-1)*q[k-1];
            q[k+(m-1)] = -z*q[k+m]; // (2*m-1)*z*q[k-1];
            k += m+1; 
        }

        /* Compute Qlm */
        k = 3; // Base index to traverse the Qlm. Initial index for q[lm] starts at l=2
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

        // pre-multiplies the Qlm with the prefactors because there's no need to do it twice below   
        /*     
        for (k=0; k<(l_max+1)*(l_max+2)/2; ++k) {
            q[k]*=prefactors[k];
        }
        */
        // Commented. We need an intact q for the derivatives. See workaround below to avoid the double multiplication.

        c[0] = 1.0;
        s[0] = 0.0;
        for (int m = 1; m < l_max+1; m++) {
            c[m] = c[m-1]*x-s[m-1]*y;
            s[m] = c[m-1]*y+s[m-1]*x;
        }

        // We fill the (cartesian) sph by combining Qlm and sine/cosine phi-dependent factors
        k = 0;
        for (int l=0; l<l_max+1; l++) {
            double pq = q[k]*prefactors[k];
            sph_i[l] = pq*M_SQRT1_2;
            for (int m=1; m<l+1; m++) {
                pq = q[k+m]*prefactors[k+m];
                sph_i[l-m] = pq*s[m];
                sph_i[l+m] = pq*c[m];
            }             
            k += l+1;
            sph_i += 2*l+1;
        }

        if (dsph != NULL) {
            double *dsph_i = dsph+i_sample*3*(l_max+1)*(l_max+1); 

            // Derivatives of q
            dqdz[0] = dqdy[0] = dqdx[0] = 0.0;  // l = m = 0
            k=1;
            for (int l = 1; l < l_max+1; l++) {
                dqdx[k+l] = 0.0;
                dqdy[k+l] = 0.0;
                dqdz[k+l] = 0.0;
                dqdx[k+l-1] = 0.0;
                dqdy[k+l-1] = 0.0;
                // dqdz[k+l-1] = 0.0;  // Not necessary
                for (int m = 0; m < l; m++) {
                    if (m != l-1) {
                        dqdx[k+m] = x*q[k-l+m+1];
                        dqdy[k+m] = y*q[k-l+m+1];
                    }
                    dqdz[k+m] = (l+m)*q[k-l+m];
                }
                k+=l+1;
            }

            // Derivatives of c, s
            dsdx[0] = 0.0;
            dcdx[0] = 0.0;
            dsdy[0] = 0.0;
            dcdy[0] = 0.0;
            for (int m = 1; m < l_max+1; m++) {
                dsdx[m] = s[m-1];
                dcdx[m] = c[m-1];
                dsdy[m] = c[m-1];
                dcdy[m] = -s[m-1];
            }

            // Chain rule:
            k=0;
            for (int l=0; l<l_max+1; l++) {
                dsph_i[l] = prefactors[k]*dqdx[k];
                for (int m=1; m<l+1; m++) {
                    dsph_i[l-m] = prefactors[k+m]*(dqdx[k+m]*s[m]+q[k+m]*dsdx[m]);;
                    dsph_i[l+m] = prefactors[k+m]*(dqdx[k+m]*c[m]+q[k+m]*dcdx[m]);;
                }
                k+=l+1;
                dsph_i += 2*l+1;
            }
            k=0;
            for (int l=0; l<l_max+1; l++) {
                dsph_i[l] = prefactors[k]*dqdy[k];
                for (int m=1; m<l+1; m++) {
                    dsph_i[l-m] = prefactors[k+m]*(dqdy[k+m]*s[m]+q[k+m]*dsdy[m]);;
                    dsph_i[l+m] = prefactors[k+m]*(dqdy[k+m]*c[m]+q[k+m]*dcdy[m]);;
                }
                k+=l+1;
                dsph_i += 2*l+1;
            }
            k=0;
            for (int l=0; l<l_max+1; l++) {
                dsph_i[l] = prefactors[k]*dqdy[k];
                for (int m=1; m<l+1; m++) {
                    dsph_i[l-m] = prefactors[k+m]*dqdz[k+m]*s[m];
                    dsph_i[l+m] = prefactors[k+m]*dqdz[k+m]*c[m];
                }
                k+=l+1;
                dsph_i += 2*l+1;
            }    

/*
                dsph_i[(l_max+1)*(l_max+1)*1+l*l+l] = prefactors[l*(l+1)/2]*dqdy[l*(l+1)/2];
                dsph_i[(l_max+1)*(l_max+1)*2+l*l+l] = prefactors[l*(l+1)/2]*dqdz[l*(l+1)/2];
                for (int m=1; m<l+1; m++) {
                    dsph_i[(l_max+1)*(l_max+1)*0+l*l+l-m] = prefactors[l*(l+1)/2+m]*(dqdx[l*(l+1)/2+m]*s[m]+q[l*(l+1)/2+m]*dsdx[m]);
                    dsph_i[(l_max+1)*(l_max+1)*1+l*l+l-m] = prefactors[l*(l+1)/2+m]*(dqdy[l*(l+1)/2+m]*s[m]+q[l*(l+1)/2+m]*dsdy[m]);
                    dsph_i[(l_max+1)*(l_max+1)*2+l*l+l-m] = prefactors[l*(l+1)/2+m]*dqdz[l*(l+1)/2+m]*s[m];
                    dsph_i[(l_max+1)*(l_max+1)*0+l*l+l+m] = prefactors[l*(l+1)/2+m]*(dqdx[l*(l+1)/2+m]*c[m]+q[l*(l+1)/2+m]*dcdx[m]);
                    dsph_i[(l_max+1)*(l_max+1)*1+l*l+l+m] = prefactors[l*(l+1)/2+m]*(dqdy[l*(l+1)/2+m]*c[m]+q[l*(l+1)/2+m]*dcdy[m]);
                    dsph_i[(l_max+1)*(l_max+1)*2+l*l+l+m] = prefactors[l*(l+1)/2+m]*dqdz[l*(l+1)/2+m]*c[m];
                }
            }
            */
        }     
    }

    if (dsph != NULL) {
        free(dqdx);
        free(dcdx);
        free(dsdx);
        free(dqdy);
        free(dcdy);
        free(dsdy);
        free(dqdz);
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
    int k; // utility index
        
    #pragma omp for
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        double x = xyz[i_sample*3+0];
        double y = xyz[i_sample*3+1];
        double z = xyz[i_sample*3+2];
        double r_sq = x*x+y*y+z*z;
        // pointer to the segment that should store the i_sample sph
        double *sph_i = sph+i_sample*(l_max+1)*(l_max+1); 

        q[0+0] = 1.0;
        k = 1;
        for (int m = 1; m < l_max+1; m++) {
            q[k+m] = -(2*m-1)*q[k-1];
            q[k+(m-1)] = -z*q[k+m]; // (2*m-1)*z*q[k-1];
            k += m+1; 
        }

        /* Compute Qlm */
        k = 3; // Base index to traverse the Qlm. Initial index for q[lm] starts at l=2
        for (int l=2; l < l_max+1; ++l) {
            double twolz = (2*l-1)*z;
            double lmrsq = (l-2)*r_sq;
            for (int m=0; m < l-1; ++m) {
                lmrsq += r_sq; // this computes (l-1+m) r_sq
                q[k+m] = (twolz*q[k-l+m]-lmrsq*q[k-(2*l-1)+m])/(l-m);
            }
            k += l+1;
        }

        // pre-multiplies the Qlm with the prefactors because there's no need to do it twice below        
        for (k=0; k<(l_max+1)*(l_max+2)/2; ++k) {
            q[k]*=prefactors[k];
        }

        c[0] = 1.0;
        s[0] = 0.0;
        for (int m = 1; m < l_max+1; m++) {
            c[m] = c[m-1]*x-s[m-1]*y;
            s[m] = c[m-1]*y+s[m-1]*x;
        }

        // We fill the (cartesian) sph by combining Qlm and sine/cosine phi-dependent factors
        k = 0;
        for (int l=0; l<l_max+1; l++) {
            sph_i[l] = q[k+0]*M_SQRT1_2;
            for (int m=1; m<l+1; m++) {
                sph_i[l-m] = q[k+m]*s[m];
                sph_i[l+m] = q[k+m]*c[m];
            }             
            k += l+1;
            sph_i += 2*l+1;
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
