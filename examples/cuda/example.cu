/** @file example.cu
 *  @brief Usage example for the CUDA C++ API
 */

#include "sphericart_cuda.hpp"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/*host macro that checks for errors in CUDA calls, and prints the file + line
 * and error string if one occurs
 */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(cudaStatus) << std::endl;                              \
            cudaDeviceReset();                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
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
    // memory, one also can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<double>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<double>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph = std::vector<double>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // the class is templated, so one can also use 32-bit float operations
    auto xyz_f = std::vector<float>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_f[i] = (float)xyz[i];
    }
    auto sph_f = std::vector<float>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<float>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_f = std::vector<float>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    sphericart::cuda::SphericalHarmonics<double> calculator_cuda(l_max);

    double* xyz_cuda;
    CUDA_CHECK(cudaMalloc(&xyz_cuda, n_samples * 3 * sizeof(double)));
    CUDA_CHECK(
        cudaMemcpy(xyz_cuda, xyz.data(), n_samples * 3 * sizeof(double), cudaMemcpyHostToDevice)
    );
    double* sph_cuda;
    CUDA_CHECK(cudaMalloc(&sph_cuda, n_samples * (l_max + 1) * (l_max + 1) * sizeof(double)));

    calculator_cuda.compute(xyz_cuda, n_samples, sph_cuda); // no gradients

    CUDA_CHECK(cudaMemcpy(
        sph.data(), sph_cuda, n_samples * (l_max + 1) * (l_max + 1) * sizeof(double), cudaMemcpyDeviceToHost
    ));

    // float version
    sphericart::cuda::SphericalHarmonics<float> calculator_cuda_f(l_max);

    float* xyz_cuda_f;
    CUDA_CHECK(cudaMalloc(&xyz_cuda_f, n_samples * 3 * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(xyz_cuda_f, xyz_f.data(), n_samples * 3 * sizeof(float), cudaMemcpyHostToDevice)
    );
    float* sph_cuda_f;
    CUDA_CHECK(cudaMalloc(&sph_cuda_f, n_samples * (l_max + 1) * (l_max + 1) * sizeof(float)));

    calculator_cuda_f.compute(xyz_cuda_f, n_samples, sph_cuda_f); // no gradients

    CUDA_CHECK(cudaMemcpy(
        sph_f.data(),
        sph_cuda_f,
        n_samples * (l_max + 1) * (l_max + 1) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    /* ===== check results ===== */

    double sph_error = 0.0, sph_norm = 0.0;
    for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
        sph_error += (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
        sph_norm += sph[i] * sph[i];
    }
    printf("Float vs double relative error: %12.8e\n", sqrt(sph_error / sph_norm));

    return 0;
}
