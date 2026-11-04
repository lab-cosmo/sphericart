/*
    Tests for the asynchronous HIP API of Sphericart:
    `compute_with_gradients_async` must only enqueue the kernel without
    synchronizing the stream, and `release_device_resources` must free the
    lazily allocated device memory (and be idempotent).

    This test uses the HIP runtime linked by the HIP backend, so building it
    requires a ROCm installation but no HIP compiler.
*/
#include <algorithm>
#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "sphericart_hip.hpp"

#define CHECK_HIP(call)                                                       \
    do {                                                                      \
        hipError_t status_ = (call);                                          \
        if (status_ != hipSuccess) {                                          \
            throw std::runtime_error(                                         \
                std::string(__FILE__) + ":" + std::to_string(__LINE__)        \
                + ": " + hipGetErrorString(status_));                         \
        }                                                                     \
    } while (0)

int main() {
    int device_count = 0;
    hipError_t status = hipGetDeviceCount(&device_count);
    if (status != hipSuccess || device_count == 0) {
        std::cout << "No usable HIP device; skipping async HIP API test" << std::endl;
        return 0;
    }

    constexpr size_t l_max = 4;
    constexpr size_t n_samples = 200000;
    constexpr size_t n_spherical = (l_max + 1) * (l_max + 1);
    constexpr int n_launches = 64;
    const size_t xyz_size = 3 * n_samples;
    const size_t sph_size = n_samples * n_spherical;
    const size_t dsph_size = 3 * sph_size;

    std::vector<double> xyz(xyz_size);
    for (size_t sample = 0; sample < n_samples; ++sample) {
        xyz[3 * sample] = 0.25 + 0.000001 * static_cast<double>(sample);
        xyz[3 * sample + 1] = -0.5 + 0.000002 * static_cast<double>(sample);
        xyz[3 * sample + 2] = 0.75 - 0.0000015 * static_cast<double>(sample);
    }
    std::vector<double> reference_sph(sph_size);
    std::vector<double> reference_dsph(dsph_size);
    std::vector<double> async_sph(sph_size);
    std::vector<double> async_dsph(dsph_size);

    double* device_xyz = nullptr;
    double* device_sph = nullptr;
    double* device_dsph = nullptr;
    hipStream_t stream = nullptr;

    try {
        CHECK_HIP(hipSetDevice(0));
        CHECK_HIP(hipStreamCreate(&stream));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_xyz), xyz_size * sizeof(double)));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_sph), sph_size * sizeof(double)));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_dsph), dsph_size * sizeof(double)));
        CHECK_HIP(
            hipMemcpy(device_xyz, xyz.data(), xyz_size * sizeof(double), hipMemcpyHostToDevice));

        using Calculator = sphericart::hip::SphericalHarmonics<double>;
        Calculator calculator(l_max);

        // synchronous reference run; this also warms up hipRTC compilation and
        // the lazy allocation of the prefactors on the device
        calculator.compute_with_gradients(
            device_xyz, n_samples, device_sph, device_dsph, static_cast<void*>(stream)
        );
        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipMemcpy(
            reference_sph.data(), device_sph, sph_size * sizeof(double), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(
            reference_dsph.data(), device_dsph, dsph_size * sizeof(double), hipMemcpyDeviceToHost));

        // a burst of async launches must enqueue without synchronizing
        const auto start = std::chrono::steady_clock::now();
        for (int launch = 0; launch < n_launches; ++launch) {
            calculator.compute_with_gradients_async(
                device_xyz, n_samples, device_sph, device_dsph, static_cast<void*>(stream)
            );
        }
        const auto enqueue_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipMemcpy(
            async_sph.data(), device_sph, sph_size * sizeof(double), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(
            async_dsph.data(), device_dsph, dsph_size * sizeof(double), hipMemcpyDeviceToHost));

        // async results must match the synchronous path bit-for-bit
        if (!std::equal(async_sph.begin(), async_sph.end(), reference_sph.begin()) ||
            !std::equal(async_dsph.begin(), async_dsph.end(), reference_dsph.begin())) {
            throw std::runtime_error("async HIP results do not match the synchronous path");
        }

        // enqueueing 64 launches must be faster than running them: if every
        // call had synchronized, this loop would have taken at least the
        // total kernel execution time.
        const auto reference_run = std::chrono::steady_clock::now();
        for (int launch = 0; launch < n_launches; ++launch) {
            calculator.compute_with_gradients(
                device_xyz, n_samples, device_sph, device_dsph, static_cast<void*>(stream)
            );
        }
        const auto sync_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - reference_run);
        if (enqueue_elapsed >= sync_elapsed) {
            throw std::runtime_error(
                "async HIP launches did not return before the stream drained "
                "(enqueue " + std::to_string(enqueue_elapsed.count()) + "ms vs sync " +
                std::to_string(sync_elapsed.count()) + "ms)");
        }

        // release_device_resources must be idempotent and transparently
        // re-allocate on the next computation
        calculator.release_device_resources();
        calculator.release_device_resources();
        calculator.compute_with_gradients(
            device_xyz, n_samples, device_sph, device_dsph, static_cast<void*>(stream)
        );
        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipMemcpy(
            async_sph.data(), device_sph, sph_size * sizeof(double), hipMemcpyDeviceToHost));
        if (!std::equal(async_sph.begin(), async_sph.end(), reference_sph.begin())) {
            throw std::runtime_error(
                "results changed after release_device_resources and re-allocation");
        }
    } catch (const std::exception& error) {
        std::cout << "async HIP API test failed: " << error.what() << std::endl;
        if (stream != nullptr) {
            static_cast<void>(hipStreamDestroy(stream));
        }
        return 1;
    }

    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(device_xyz));
    CHECK_HIP(hipFree(device_sph));
    CHECK_HIP(hipFree(device_dsph));

    std::cout << "Test passed" << std::endl;
    return 0;
}
