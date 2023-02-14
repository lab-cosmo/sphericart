#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include "sphericart.h"

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
        cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, NULL); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    }
    printf("Call without derivatives took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    double *dsph = (double*) malloc(sizeof(double)*n_samples*3*(l_max+1)*(l_max+1));
    
    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, dsph); 
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
        cartesian_spherical_harmonics_hybrid(n_samples, l_max, prefactors, xyz, sph, dsph); 
        gettimeofday(&end, NULL);
        time = (end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6)/n_samples;
        time_total += time; time2_total +=  time*time;
    } 
    printf("Call with derivatives (hybrid) took %f ± %f µs/sample\n", 
            1e6*time_total/n_tries, 1e6*sqrt(time2_total/n_tries - 
                                       (time_total/n_tries)*(time_total/n_tries))
            );

    double *sph1 = (double*) malloc(sizeof(double)*n_samples*(l_max+1)*(l_max+1));
    double *dsph1 = (double*) malloc(sizeof(double)*n_samples*3*(l_max+1)*(l_max+1));
    
    printf("================ Low-l timings ===========\n");

    time_total = time2_total = 0;
    for (int i_try = 0; i_try < n_tries; i_try++) {
        gettimeofday(&start, NULL);
        cartesian_spherical_harmonics(n_samples, 1, prefactors, xyz, sph, NULL); 
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
        cartesian_spherical_harmonics(n_samples, 1, prefactors, xyz, sph, dsph); 
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
        cartesian_spherical_harmonics(n_samples, 2, prefactors, xyz, sph, NULL); 
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
        cartesian_spherical_harmonics(n_samples, 2, prefactors, xyz, sph, dsph); 
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
        cartesian_spherical_harmonics(n_samples, 3, prefactors, xyz, sph, NULL); 
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
        cartesian_spherical_harmonics(n_samples, 3, prefactors, xyz, sph, dsph); 
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
        cartesian_spherical_harmonics(n_samples, 4, prefactors, xyz, sph, NULL); 
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
        cartesian_spherical_harmonics(n_samples, 4, prefactors, xyz, sph, dsph); 
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

    int k=0; l_max=4;
    for (int l=0; l<(l_max+1); l++)
    {
        for (int m=-l; m<(l+1); m++) {
            printf("L= %d   m= %d  \n", l, m);
            if (fabs(sph[k+l-m]/sph1[k+l-m]-1)>1e-6) printf("!!!! ");
            printf("SPH: %e, %e\n", sph[k+l+m], sph1[k+l+m]);
            if (fabs(dsph[k+l+m]/dsph1[k+l+m]-1)>1e-6) printf("!!!! ");
            printf("DxSPH: %e, %e\n", dsph[k+l+m], dsph1[k+l+m]);
            if (fabs(dsph[(l_max+1)*(l_max+1)+k+l+m]/dsph1[(l_max+1)*(l_max+1)+k+l+m]-1)>1e-6) printf("!!!! ");
            printf("DySPH: %e, %e\n", dsph[(l_max+1)*(l_max+1)+k+l+m], dsph1[(l_max+1)*(l_max+1)+k+l+m]);
            if (fabs(dsph[(l_max+1)*(l_max+1)*2+k+l+m]/dsph1[(l_max+1)*(l_max+1)*2+k+l+m]-1)>1e-6) printf("!!!! ");
            printf("DzSPH: %e, %e\n", dsph[(l_max+1)*(l_max+1)*2+k+l+m], dsph1[(l_max+1)*(l_max+1)*2+k+l+m]);
        }
        k+=2*l+1;            
        //printf("chk %e, %e\n", dsph[i], dsph1[i]);
    }


    free(xyz);
    free(prefactors);
    free(sph);
    free(dsph);

    return 0;
}
