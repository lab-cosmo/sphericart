#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include "sphericart.h"

#define _SPH_TOL 1e-5

int main(int argc, char *argv[]) {

    unsigned int n_samples = 10000;
    unsigned int n_tries = 100;
    unsigned int l_max = 10;
    int c;

    // parse command line options
    while ((c = getopt (argc, argv, "l:s:t:")) != -1) {
        switch (c) {
        case 'l':
            sscanf(optarg, "%u", &l_max);
            break;
        case 's':
            sscanf(optarg, "%u", &n_samples);
            break;
        case 't':
            sscanf(optarg, "%u", &n_tries);
            break;
        case '?':
            if (optopt == 'c')
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
            return 1;
        default:
            abort ();
        }
    }

    printf("Running with l_max= %d, n_tries= %d, n_samples= %d\n", l_max, n_tries, n_samples);
    printf("\n");
    double *prefactors = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
    compute_sph_prefactors(l_max, prefactors);

    double *xyz = (double*) malloc(sizeof(double)*n_samples*3);
    for (int i=0; i<n_samples*3; ++i) {
        xyz[i] = (double)rand()/ (double) RAND_MAX *2.0-1.0;
    }

    double *sph = (double*) malloc(sizeof(double)*n_samples*(l_max+1)*(l_max+1));

    struct timeval start, end;
    double time, time_total, time2_total;

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, l_max, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    }
    printf("Call without derivatives (generic) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    double *dsph = (double*) malloc(sizeof(double)*n_samples*3*(l_max+1)*(l_max+1));
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, l_max, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with derivatives (generic) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    double *sph1 = (double*) malloc(sizeof(double)*n_samples*(l_max+1)*(l_max+1));
    double *dsph1 = (double*) malloc(sizeof(double)*n_samples*3*(l_max+1)*(l_max+1));
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call without derivatives took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with derivatives took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        normalized_cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call without derivatives (normalized) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        normalized_cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with derivatives (normalized) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    cartesian_spherical_harmonics_generic(n_samples, l_max, prefactors, xyz, sph, dsph);
    cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph1, dsph1); 

    int size3 = 3*(l_max+1)*(l_max+1);  // Size of the third dimension in derivative arrays (or second in normal sph arrays).
    int size2 = (l_max+1)*(l_max+1);  // Size of the second+third dimensions in derivative arrays
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        for (int l=0; l<(l_max+1); l++) {
            for (int m=-l; m<=l; m++) {
                if (fabs(sph[size2*i_sample+l*l+l+m]/sph1[size2*i_sample+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);
                    printf("SPH: %e, %e\n", sph[size2*i_sample+l*l+l+m], sph1[size2*i_sample+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*0+l*l+l+m]/dsph1[size3*i_sample+size2*0+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);           
                    printf("DxSPH: %e, %e\n", dsph[size3*i_sample+size2*0+l*l+l+m], dsph1[size3*i_sample+size2*0+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*1+l*l+l+m]/dsph1[size3*i_sample+size2*1+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);            
                    printf("DySPH: %e, %e\n", dsph[size3*i_sample+size2*1+l*l+l+m],dsph1[size3*i_sample+size2*1+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*2+l*l+l+m]/dsph1[size3*i_sample+size2*2+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);         
                    printf("DzSPH: %e, %e\n", dsph[size3*i_sample+size2*2+l*l+l+m], dsph1[size3*i_sample+size2*2+l*l+l+m]);
                }
            }
        }
    }

    printf("\n");
    printf("================ Low-l timings ===========\n");

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 1, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=1 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 1, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=1 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l1(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=1 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l1(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=1 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 2, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=2 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 2, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=2 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l2(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=2 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l2(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=2 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 3, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=3 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 3, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=3 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l3(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=3 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l3(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=3 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 4, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=4 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 4, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=4 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l4(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=4 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l4(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=4 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 5, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=5 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 5, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=5 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l5(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=5 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l5(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=5 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 6, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=6 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_generic(n_samples, 6, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with l=6 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l6(n_samples, xyz, sph1, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=6 (sph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics_l6(n_samples, xyz, sph1, dsph1); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with hardcoded l=6 (dsph) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    l_max=6;
    size3 = 3*(l_max+1)*(l_max+1);  // Size of the third dimension in derivative arrays (or second in normal sph arrays).
    size2 = (l_max+1)*(l_max+1);  // Size of the second+third dimensions in derivative arrays
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        for (int l=0; l<(l_max+1); l++) {
            for (int m=-l; m<=l; m++) {
                if (fabs(sph[size2*i_sample+l*l+l+m]/sph1[size2*i_sample+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);
                    printf("SPH: %e, %e\n", sph[size2*i_sample+l*l+l+m], sph1[size2*i_sample+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*0+l*l+l+m]/dsph1[size3*i_sample+size2*0+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);           
                    printf("DxSPH: %e, %e\n", dsph[size3*i_sample+size2*0+l*l+l+m], dsph1[size3*i_sample+size2*0+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*1+l*l+l+m]/dsph1[size3*i_sample+size2*1+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);            
                    printf("DySPH: %e, %e\n", dsph[size3*i_sample+size2*1+l*l+l+m],dsph1[size3*i_sample+size2*1+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*2+l*l+l+m]/dsph1[size3*i_sample+size2*2+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %d, L = %d, m = %d \n", i_sample, l, m);
                    printf("DzSPH: %e, %e\n", dsph[size3*i_sample+size2*2+l*l+l+m], dsph1[size3*i_sample+size2*2+l*l+l+m]);
                }
            }
        }
    }

    free(xyz);
    free(prefactors);    
    free(sph);
    free(dsph);
    free(sph1);
    free(dsph1);

    return 0;
}
