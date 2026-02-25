/** @file benchmark_sycl.cpp
 *  @brief benchmarks for the SYCL (GPU) API
 *
 * Compares cost of evaluation with and without gradients/hessians
 * for the SYCL implementation
 */

#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sphericart_sycl.hpp"
#include "sphericart.hpp"

#ifndef DTYPE
#define DTYPE double
#endif

using namespace sphericart::sycl;

template <typename Fn>
inline void benchmark(
    std::string context, size_t n_samples, size_t n_tries, Fn function, ::sycl::queue& q
) {
    // warmup
    for (size_t i_try = 0; i_try < 10; i_try++) {
        function();
        q.wait();
    }

    auto time = 0.0;
    auto time2 = 0.0;

    for (size_t i_try = 0; i_try < n_tries; i_try++) {
        auto start = std::chrono::steady_clock::now();

        function();
        q.wait(); // Ensure kernel completes before timing

        auto end = std::chrono::steady_clock::now();

        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time += duration / n_samples;
        time2 += duration * duration / (n_samples * n_samples);
    }

    auto std_dev = sqrt(time2 / n_tries - (time / n_tries) * (time / n_tries));
    std::cout << context << " took " << std::fixed << std::setprecision(2) << time / n_tries;
    std::cout << " ± " << std_dev << " ns / sample" << std::endl;
}

template <typename DTYPE_T>
void run_timings(int l_max, int n_tries, int n_samples, ::sycl::queue& q) {
    // Initialize random xyz coordinates on host
    auto xyz_host = std::vector<DTYPE_T>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_host[i] = (DTYPE_T)rand() / (DTYPE_T)RAND_MAX * 2.0 - 1.0;
    }

    // Allocate device memory
    DTYPE_T* xyz_device = ::sycl::malloc_device<DTYPE_T>(n_samples * 3, q);
    DTYPE_T* sph_device = ::sycl::malloc_device<DTYPE_T>(n_samples * (l_max + 1) * (l_max + 1), q);
    DTYPE_T* dsph_device =
        ::sycl::malloc_device<DTYPE_T>(n_samples * 3 * (l_max + 1) * (l_max + 1), q);
    DTYPE_T* ddsph_device =
        ::sycl::malloc_device<DTYPE_T>(n_samples * 9 * (l_max + 1) * (l_max + 1), q);

    // Copy xyz to device
    q.memcpy(xyz_device, xyz_host.data(), sizeof(DTYPE_T) * n_samples * 3).wait();

    // Create SYCL calculator
    SphericalHarmonics<DTYPE_T> calculator_sycl(l_max);

    std::cout << "\n=== SYCL SphericalHarmonics (normalized) ===" << std::endl;

    benchmark(
        "SYCL: Values only         ",
        n_samples,
        n_tries,
        [&]() { calculator_sycl.compute(xyz_device, n_samples, sph_device); },
        q
    );

    benchmark(
        "SYCL: Values + gradients  ",
        n_samples,
        n_tries,
        [&]() {
            calculator_sycl.compute_with_gradients(xyz_device, n_samples, sph_device, dsph_device);
        },
        q
    );

    benchmark(
        "SYCL: Values + grad + hess",
        n_samples,
        n_tries,
        [&]() {
            calculator_sycl.compute_with_hessians(
                xyz_device, n_samples, sph_device, dsph_device, ddsph_device
            );
        },
        q
    );

    // Create SYCL SolidHarmonics calculator
    SolidHarmonics<DTYPE_T> solid_calculator_sycl(l_max);

    std::cout << "\n=== SYCL SolidHarmonics (unnormalized) ===" << std::endl;

    benchmark(
        "SYCL: Values only         ",
        n_samples,
        n_tries,
        [&]() { solid_calculator_sycl.compute(xyz_device, n_samples, sph_device); },
        q
    );

    benchmark(
        "SYCL: Values + gradients  ",
        n_samples,
        n_tries,
        [&]() {
            solid_calculator_sycl.compute_with_gradients(
                xyz_device, n_samples, sph_device, dsph_device
            );
        },
        q
    );

    benchmark(
        "SYCL: Values + grad + hess",
        n_samples,
        n_tries,
        [&]() {
            solid_calculator_sycl.compute_with_hessians(
                xyz_device, n_samples, sph_device, dsph_device, ddsph_device
            );
        },
        q
    );

    // Compare with CPU implementation
    std::cout << "\n=== CPU Comparison ===" << std::endl;

    auto sph_cpu = std::vector<DTYPE_T>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_cpu = std::vector<DTYPE_T>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_cpu = std::vector<DTYPE_T>(n_samples * 9 * (l_max + 1) * (l_max + 1), 0.0);

    sphericart::SphericalHarmonics<DTYPE_T> calculator_cpu(l_max);

    // CPU benchmark (using a dummy queue just for the interface)
    auto cpu_time = 0.0;
    auto cpu_time2 = 0.0;

    // warmup
    for (size_t i_try = 0; i_try < 10; i_try++) {
        calculator_cpu.compute(xyz_host, sph_cpu);
    }

    for (size_t i_try = 0; i_try < n_tries; i_try++) {
        auto start = std::chrono::steady_clock::now();
        calculator_cpu.compute(xyz_host, sph_cpu);
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cpu_time += duration / n_samples;
        cpu_time2 += duration * duration / (n_samples * n_samples);
    }
    auto cpu_std = sqrt(cpu_time2 / n_tries - (cpu_time / n_tries) * (cpu_time / n_tries));
    std::cout << "CPU:  Values only          took " << std::fixed << std::setprecision(2)
              << cpu_time / n_tries << " ± " << cpu_std << " ns / sample" << std::endl;

    // CPU with gradients
    cpu_time = 0.0;
    cpu_time2 = 0.0;
    for (size_t i_try = 0; i_try < 10; i_try++) {
        calculator_cpu.compute_with_gradients(xyz_host, sph_cpu, dsph_cpu);
    }
    for (size_t i_try = 0; i_try < n_tries; i_try++) {
        auto start = std::chrono::steady_clock::now();
        calculator_cpu.compute_with_gradients(xyz_host, sph_cpu, dsph_cpu);
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cpu_time += duration / n_samples;
        cpu_time2 += duration * duration / (n_samples * n_samples);
    }
    cpu_std = sqrt(cpu_time2 / n_tries - (cpu_time / n_tries) * (cpu_time / n_tries));
    std::cout << "CPU:  Values + gradients   took " << std::fixed << std::setprecision(2)
              << cpu_time / n_tries << " ± " << cpu_std << " ns / sample" << std::endl;

    // Verify correctness by comparing CPU and GPU results
    std::cout << "\n=== Correctness Verification ===" << std::endl;

    // Compute on CPU
    calculator_cpu.compute(xyz_host, sph_cpu);

    // Compute on GPU and copy back
    calculator_sycl.compute(xyz_device, n_samples, sph_device);
    q.wait();

    auto sph_gpu = std::vector<DTYPE_T>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    q.memcpy(sph_gpu.data(), sph_device, sizeof(DTYPE_T) * n_samples * (l_max + 1) * (l_max + 1))
        .wait();

    // Calculate relative error
    DTYPE_T error = 0.0, norm = 0.0;
    for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
        error += (sph_cpu[i] - sph_gpu[i]) * (sph_cpu[i] - sph_gpu[i]);
        norm += sph_cpu[i] * sph_cpu[i];
    }
    std::cout << "CPU vs GPU relative error: " << std::scientific << std::setprecision(8)
              << sqrt(error / norm) << std::endl;

    // Free device memory
    ::sycl::free(xyz_device, q);
    ::sycl::free(sph_device, q);
    ::sycl::free(dsph_device, q);
    ::sycl::free(ddsph_device, q);
}

template <typename DTYPE_T>
void run_lmax_sweep(int max_l, int n_tries, int n_samples, ::sycl::queue& q) {
    std::cout << "\n========== L_max Sweep ==========" << std::endl;
    std::cout << "L_max\tValues (ns)\tGradients (ns)" << std::endl;
    std::cout << "-----\t-----------\t--------------" << std::endl;

    for (int l_max = 1; l_max <= max_l; ++l_max) {
        // Allocate device memory
        DTYPE_T* xyz_device = ::sycl::malloc_device<DTYPE_T>(n_samples * 3, q);
        DTYPE_T* sph_device =
            ::sycl::malloc_device<DTYPE_T>(n_samples * (l_max + 1) * (l_max + 1), q);
        DTYPE_T* dsph_device =
            ::sycl::malloc_device<DTYPE_T>(n_samples * 3 * (l_max + 1) * (l_max + 1), q);

        // Initialize random xyz coordinates on host and copy to device
        auto xyz_host = std::vector<DTYPE_T>(n_samples * 3, 0.0);
        for (size_t i = 0; i < n_samples * 3; ++i) {
            xyz_host[i] = (DTYPE_T)rand() / (DTYPE_T)RAND_MAX * 2.0 - 1.0;
        }
        q.memcpy(xyz_device, xyz_host.data(), sizeof(DTYPE_T) * n_samples * 3).wait();

        SphericalHarmonics<DTYPE_T> calculator(l_max);

        // Warmup
        for (int i = 0; i < 10; ++i) {
            calculator.compute(xyz_device, n_samples, sph_device);
            q.wait();
        }

        // Benchmark values only
        double time_values = 0.0;
        for (int i_try = 0; i_try < n_tries; ++i_try) {
            auto start = std::chrono::steady_clock::now();
            calculator.compute(xyz_device, n_samples, sph_device);
            q.wait();
            auto end = std::chrono::steady_clock::now();
            time_values += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        time_values /= (n_tries * n_samples);

        // Warmup gradients
        for (int i = 0; i < 10; ++i) {
            calculator.compute_with_gradients(xyz_device, n_samples, sph_device, dsph_device);
            q.wait();
        }

        // Benchmark with gradients
        double time_grads = 0.0;
        for (int i_try = 0; i_try < n_tries; ++i_try) {
            auto start = std::chrono::steady_clock::now();
            calculator.compute_with_gradients(xyz_device, n_samples, sph_device, dsph_device);
            q.wait();
            auto end = std::chrono::steady_clock::now();
            time_grads += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        time_grads /= (n_tries * n_samples);

        std::cout << l_max << "\t" << std::fixed << std::setprecision(2) << time_values << "\t\t"
                  << time_grads << std::endl;

        ::sycl::free(xyz_device, q);
        ::sycl::free(sph_device, q);
        ::sycl::free(dsph_device, q);
    }
}

int main(int argc, char* argv[]) {
    size_t n_samples = 10000;
    size_t n_tries = 100;
    size_t l_max = 10;
    bool run_sweep = false;

    // parse command line options
    int c;
    while ((c = getopt(argc, argv, "l:s:t:w")) != -1) {
        switch (c) {
        case 'l':
            sscanf(optarg, "%zu", &l_max);
            break;
        case 's':
            sscanf(optarg, "%zu", &n_samples);
            break;
        case 't':
            sscanf(optarg, "%zu", &n_tries);
            break;
        case 'w':
            run_sweep = true;
            break;
        case '?':
            if (optopt == 'c') {
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            } else if (isprint(optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
            return 1;
        default:
            abort();
        }
    }

    // Initialize SYCL queue
    ::sycl::queue q;

    std::cout << "========================================" << std::endl;
    std::cout << "SYCL Spherical Harmonics Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "n_samples: " << n_samples << std::endl;
    std::cout << "n_tries: " << n_tries << std::endl;
    std::cout << "l_max: " << l_max << std::endl;
    std::cout << "========================================" << std::endl;

    if (sizeof(DTYPE) == 4) {
        std::cout << "\n****************** SINGLE PRECISION ******************" << std::endl;
        run_timings<float>(l_max, n_tries, n_samples, q);
        if (run_sweep) {
            run_lmax_sweep<float>(l_max, n_tries, n_samples, q);
        }
    } else {
        std::cout << "\n****************** DOUBLE PRECISION ******************" << std::endl;
        run_timings<double>(l_max, n_tries, n_samples, q);
        if (run_sweep) {
            run_lmax_sweep<double>(l_max, n_tries, n_samples, q);
        }
    }

    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}
