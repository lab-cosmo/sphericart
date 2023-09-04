/** @file test_hardcoding.cpp
 *  @brief Checks consistency of generic and hardcoded implementations
 */

#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iostream>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

#define _SPH_TOL 1e-6
#define DTYPE double
using namespace sphericart;

// shorthand for all-past-1 generic sph only
inline void compute_generic(int n_samples, int l_max, DTYPE *prefactors,
                            DTYPE *xyz, DTYPE *sph, DTYPE *dsph,
                            DTYPE *buffers) {
  if (dsph == nullptr) {
    generic_sph<DTYPE, false, false, false, 1>(
        xyz, sph, dsph, nullptr, n_samples, l_max, prefactors, buffers);
  } else {
    generic_sph<DTYPE, true, false, false, 1>(
        xyz, sph, dsph, nullptr, n_samples, l_max, prefactors, buffers);
  }
}

int main(int argc, char *argv[]) {
  size_t n_samples = 100;
  size_t l_max = SPHERICART_LMAX_HARDCODED;

  // parse command line options
  int c;
  while ((c = getopt(argc, argv, "s:")) != -1) {
    switch (c) {
    case 's':
      sscanf(optarg, "%zu", &n_samples);
      break;
    case '?':
      if (optopt == 'c')
        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
      else if (isprint(optopt))
        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
      else
        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
      return 1;
    default:
      abort();
    }
  }

  std::cout << "\n============= l_max_hardcoded = " << l_max
            << " ==============" << std::endl;

  auto *buffers =
      new DTYPE[(l_max + 1) * (l_max + 2) / 2 * 3 * omp_get_max_threads()];
  auto prefactors = std::vector<DTYPE>((l_max + 1) * (l_max + 2), 0.0);
  compute_sph_prefactors(l_max, prefactors.data());

  // random values
  auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
  for (size_t i = 0; i < n_samples * 3; ++i) {
    xyz[i] = (DTYPE)rand() / (DTYPE)RAND_MAX * 2.0 - 1.0;
  }

  auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
  auto dsph =
      std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);

  compute_generic(n_samples, l_max, prefactors.data(), xyz.data(), sph.data(),
                  dsph.data(), buffers);

  auto sph1 = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
  auto dsph1 =
      std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);

  SphericalHarmonics<DTYPE> SH(l_max, false);
  SH.compute_with_gradients(xyz, sph1, dsph1);

  int size3 = 3 * (l_max + 1) *
              (l_max + 1); // Size of the third dimension in derivative
                           // arrays (or second in normal sph arrays).
  int size2 =
      (l_max + 1) *
      (l_max + 1); // Size of the second+third dimensions in derivative arrays
  bool test_passed = true;
  for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
    for (size_t l = 0; l < (l_max + 1); l++) {
      for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
        if (fabs(sph[size2 * i_sample + l * l + l + m] /
                     sph1[size2 * i_sample + l * l + l + m] -
                 1) > _SPH_TOL) {
          printf("Mismatch detected at i_sample = %zu, L = %zu, m = "
                 "%d \n",
                 i_sample, l, m);
          printf("SPH: %e, %e\n", sph[size2 * i_sample + l * l + l + m],
                 sph1[size2 * i_sample + l * l + l + m]);
          test_passed = false;
        }
        if (fabs(dsph[size3 * i_sample + size2 * 0 + l * l + l + m] /
                     dsph1[size3 * i_sample + size2 * 0 + l * l + l + m] -
                 1) > _SPH_TOL) {
          printf("Mismatch detected at i_sample = %zu, L = %zu, m = "
                 "%d \n",
                 i_sample, l, m);
          printf("DxSPH: %e, %e\n",
                 dsph[size3 * i_sample + size2 * 0 + l * l + l + m],
                 dsph1[size3 * i_sample + size2 * 0 + l * l + l + m]);
          test_passed = false;
        }
        if (fabs(dsph[size3 * i_sample + size2 * 1 + l * l + l + m] /
                     dsph1[size3 * i_sample + size2 * 1 + l * l + l + m] -
                 1) > _SPH_TOL) {
          printf("Mismatch detected at i_sample = %zu, L = %zu, m = "
                 "%d \n",
                 i_sample, l, m);
          printf("DySPH: %e, %e\n",
                 dsph[size3 * i_sample + size2 * 1 + l * l + l + m],
                 dsph1[size3 * i_sample + size2 * 1 + l * l + l + m]);
          test_passed = false;
        }
        if (fabs(dsph[size3 * i_sample + size2 * 2 + l * l + l + m] /
                     dsph1[size3 * i_sample + size2 * 2 + l * l + l + m] -
                 1) > _SPH_TOL) {
          printf("Mismatch detected at i_sample = %zu, L = %zu, m = "
                 "%d \n",
                 i_sample, l, m);
          printf("DzSPH: %e, %e\n",
                 dsph[size3 * i_sample + size2 * 2 + l * l + l + m],
                 dsph1[size3 * i_sample + size2 * 2 + l * l + l + m]);
          test_passed = false;
        }
      }
    }
  }
  if (test_passed) {
    printf("Consistency test passed\n");
    return 0;
  } else {
    printf("Consistency test failed\n");
    return -1;
  }
}
