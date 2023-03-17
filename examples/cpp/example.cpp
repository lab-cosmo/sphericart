#include <iostream>
#include <vector>
#include "sphericart.hpp"


int main() {
    size_t l_max = 5;

    auto prefactors = std::vector<double>((l_max+1)*(l_max+2), 0.0);
    sphericart::compute_sph_prefactors(l_max, prefactors.data());

    // To be completed once the interface is finished

    return 0;
}
