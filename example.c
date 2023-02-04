#include "sphericart.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"

/* main.c */
int main(int argc, char *argv[]) {
    unsigned int n_samples = 10000;
    unsigned int n_tests = 10;
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
            sscanf(optarg, "%u", &n_tests);
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
    double *sph = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+1));
    
    cartesian_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, NULL);    
}