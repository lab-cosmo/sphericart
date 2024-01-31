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
#include <chrono>


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

template <class scalar_t> void timing() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 10000;
    size_t l_max = 0;

    // initializes samples
    auto xyz = std::vector<scalar_t>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (scalar_t)rand() / (scalar_t)RAND_MAX * 2.0 - 1.0;
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, one also can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph =
        std::vector<scalar_t>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph =
        std::vector<scalar_t>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph = std::vector<scalar_t>(
        n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    sphericart::cuda::SphericalHarmonics<scalar_t> calculator_cuda(l_max);

    scalar_t *xyz_cuda;
    CUDA_CHECK(cudaMalloc(&xyz_cuda, n_samples * 3 * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(xyz_cuda, xyz.data(),
                          n_samples * 3 * sizeof(scalar_t),
                          cudaMemcpyHostToDevice));
    scalar_t *sph_cuda;
    CUDA_CHECK(cudaMalloc(&sph_cuda, n_samples * (l_max + 1) * (l_max + 1) *
                                         sizeof(scalar_t)));

    scalar_t *dsph_cuda;
    CUDA_CHECK(cudaMalloc(&dsph_cuda, 3 * n_samples * (l_max + 1) *
                                          (l_max + 1) * sizeof(scalar_t)));
    
    auto start = std::chrono::high_resolution_clock::now();


    //calculator_cuda.compute_with_gradients(xyz_cuda, n_samples, sph_cuda, dsph_cuda); // no gradients */
    calculator_cuda.compute(xyz_cuda, n_samples, sph_cuda); // no gradients

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Print the duration in microseconds
    std::cout << "Time taken by function: " << duration.count() << " nanoseconds" << std::endl;
    std::cout << "" <<((double) duration.count()) / ((double) n_samples) << " ns/sample" << std::endl;
    // */
    CUDA_CHECK(
        cudaMemcpy(sph.data(), sph_cuda,
                   n_samples * (l_max + 1) * (l_max + 1) * sizeof(scalar_t),
                   cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++) {
        std::cout << sph[i] << std::endl;
    }
}

int main() {
    timing<float>();

    return 0;
}
