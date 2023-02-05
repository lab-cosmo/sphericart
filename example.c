#include "sphericart.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

    unsigned int n_samples = 10000;
    unsigned int n_tries = 10;
    unsigned int l_max = 10;
    int c;

    // parse command line options
    while ((c = getopt (argc, argv, "l:s:t:")) != -1) {
        switch (c) {
        case 'l':
            printf("%s ", optarg);
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

    double *prefactors = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+1));

    compute_sph_prefactors(l_max, prefactors);

    double *xyz = (double*) malloc(sizeof(double)*n_samples*3);
    double *sph = (double*) malloc(sizeof(double)*n_samples*(l_max+1)*(l_max+1));

    struct timeval start, end;
    double time_taken;

    gettimeofday(&start, NULL);
    for (int i_try = 0; i_try < n_tries; i_try++) {
        cartesian_spherical_harmonics_naive(n_samples, l_max, prefactors, xyz, sph, NULL); 
    }
    gettimeofday(&end, NULL);

    time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6; // in seconds

    printf("Naive implementation took %f ms\n", 1000.0*time_taken/n_tries);

    gettimeofday(&start, NULL);
    for (int i_try = 0; i_try < n_tries; i_try++) {
        cartesian_spherical_harmonics_cache(n_samples, l_max, prefactors, xyz, sph, NULL); 
    } 
    gettimeofday(&end, NULL);

    time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6; // in seconds

    printf("Cache implementation took %f ms\n", 1000.0*time_taken/n_tries);

    return 0;
}
