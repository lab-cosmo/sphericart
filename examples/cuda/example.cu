/** @file example.cpp
 *  @brief Usage example for the C++ API
 */

#include "sphericart_cuda.hpp"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace sphericart::cuda;

/*host macro that checks for errors in CUDA calls, and prints the file + line
 * and error string if one occurs
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t cudaStatus = (call);                                       \
        if (cudaStatus != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(cudaStatus) << std::endl; \
            cudaDeviceReset();                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 10000;
    size_t l_max = 10;

    // initializes samples
    auto xyz = std::vector<double>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, however one can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<double>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph =
        std::vector<double>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph =
        std::vector<double>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    auto calculator_cuda = sphericart::cuda::SphericalHarmonics<double>(l_max);

    double *xyz_cuda;
    CUDA_CHECK(cudaMalloc(&xyz_cuda, n_samples * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(xyz_cuda, xyz.data(), n_samples * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
    double *sph_cuda;
    CUDA_CHECK(cudaMalloc(&sph_cuda, n_samples * (l_max + 1) * (l_max + 1) *
                                         sizeof(double)));

    calculator_cuda.compute(xyz_cuda, n_samples, false, false,
                            sph_cuda); // no gradients */

    CUDA_CHECK(
        cudaMemcpy(sph.data(), sph_cuda,
                   n_samples * (l_max + 1) * (l_max + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++) {
        std::cout << sph[i] << std::endl;
    }

    return 0;
}
