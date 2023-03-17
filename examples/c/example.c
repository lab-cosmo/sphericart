#include "stdlib.h"
#include "sphericart.h"
#include "sphericart/exports.h"


int main() {
    int l_max = 5;

    double *prefactors = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
    sphericart_compute_sph_prefactors(l_max, prefactors);

    // To be done once the interface is established

    free(prefactors);

    return 0;
}
