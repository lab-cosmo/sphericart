/** @file benchmarks.cpp
 *  @brief benchmarks for the C++ (CPU) API
 *
 * Compares cost of evaluation with and without hardcoding, and with and without normalization
*/

#include <unistd.h>
#include <sys/time.h>

#include <cmath>
#include <chrono>
#include <iostream>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

#define _SPH_TOL 1e-6
using namespace sphericart;
// shorthand for all-past-1 generic sph only
template<typename DTYPE>
inline void compute_generic(int n_samples, int l_max, DTYPE *prefactors, DTYPE *xyz, DTYPE *sph, DTYPE *dsph, DTYPE* ddsph, DTYPE* buffers) {
    if (dsph==nullptr) {
        generic_sph<DTYPE, false, false, false, 1>(xyz, sph, dsph, ddsph, n_samples, l_max, prefactors, buffers);
    } else {
        generic_sph<DTYPE, true, false, false, 1>(xyz, sph, dsph, ddsph, n_samples, l_max, prefactors, buffers);
    }
}

template<typename Fn>
inline void benchmark(std::string context, size_t n_samples, size_t n_tries, Fn function) {
    // warmup
    for (size_t i_try = 0; i_try < 10; i_try++) {
        function();
    }

    auto time = 0.0;
    auto time2 = 0.0;

    for (size_t i_try = 0; i_try < n_tries; i_try++) {
        auto start = std::chrono::steady_clock::now();

        function();

        auto end = std::chrono::steady_clock::now();

        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time += duration / n_samples;
        time2 += duration * duration / (n_samples * n_samples);
    }

    auto std = sqrt(time2 / n_tries - (time / n_tries) * (time / n_tries));
    std::cout << context << " took " << time / n_tries;
    std::cout << " ± " << std << " ns / sample" << std::endl;
}

template<typename DTYPE>
void run_timings(int l_max, int n_tries, int n_samples) {
    auto *buffers = new DTYPE[ (l_max + 1) * (l_max + 2) / 2 * 3 * omp_get_max_threads()];
    auto prefactors = std::vector<DTYPE>((l_max+1)*(l_max+2), 0.0);
    compute_sph_prefactors(l_max, prefactors.data());

    auto xyz = std::vector<DTYPE>(n_samples*3, 0.0);
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz[i] = (DTYPE)rand()/ (DTYPE) RAND_MAX *2.0-1.0;
    }

    auto sph = std::vector<DTYPE>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph = std::vector<DTYPE>(n_samples*3*(l_max+1)*(l_max+1), 0.0);

    benchmark("Call without derivatives (no hardcoding)", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, l_max, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("Call with derivatives (no hardcoding)", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, l_max, prefactors.data(), xyz.data(), sph.data(), nullptr, dsph.data(), buffers);
    });
    std::cout << std::endl;

    auto sxyz = std::vector<DTYPE>(3, 0.0);
    auto ssph = std::vector<DTYPE>((l_max+1)*(l_max+1), 0.0);
    auto sdsph = std::vector<DTYPE>(3*(l_max+1)*(l_max+1), 0.0);
    auto sph1 = std::vector<DTYPE>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph1 = std::vector<DTYPE>(n_samples*3*(l_max+1)*(l_max+1), 0.0);

    {
        SphericalHarmonics<DTYPE> calculator(l_max, false);
        sxyz[0] = xyz[0]; sxyz[1] = xyz[1]; sxyz[2] = xyz[2];

        // single-sample evaluation
        benchmark("Sample without derivatives", 1, n_tries, [&](){
            calculator.compute(sxyz, ssph);
        });

        benchmark("Sample with derivatives", 1, n_tries, [&](){
            calculator.compute_with_gradients(sxyz, ssph, sdsph);
        });

        std::cout << std::endl;

        benchmark("Call without derivatives", n_samples, n_tries, [&](){
            calculator.compute(xyz, sph1);
        });

        benchmark("Call with derivatives", n_samples, n_tries, [&](){
            calculator.compute_with_gradients(xyz, sph1, dsph1);
        });
    }

    {
        SphericalHarmonics<DTYPE> calculator(l_max, true);
        benchmark("Call without derivatives (normalized)", n_samples, n_tries, [&](){
            calculator.compute(xyz, sph1);
        });

        benchmark("Call with derivatives (normalized)", n_samples, n_tries, [&](){
            calculator.compute_with_gradients(xyz, sph1, dsph1);
        });
    }
    std::cout << std::endl;

    std::cout << "================ Low-l timings ===========" << std::endl;

    compute_sph_prefactors(1, prefactors.data());
    benchmark("L=1 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 1, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=1 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 1, prefactors.data(), xyz.data(), sph.data(), dsph.data(), nullptr, buffers);
    });

    benchmark("L=1 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 1>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=1 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 1>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    if (l_max == 1) {
        free(buffers);
        return;
    }

    //========================================================================//
    compute_sph_prefactors(2, prefactors.data());
    benchmark("L=2 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 2, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=2 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 2, prefactors.data(), xyz.data(), sph.data(), nullptr, dsph.data(), buffers);
    });

    benchmark("L=2 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 2>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=2 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 2>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    if (l_max == 2) {
        free(buffers);
        return;
    }

    //========================================================================//
    compute_sph_prefactors(3, prefactors.data());
    benchmark("L=3 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 3, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=3 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 3, prefactors.data(), xyz.data(), sph.data(), dsph.data(), nullptr, buffers);
    });

    benchmark("L=3 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 3>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples, 0, nullptr, nullptr);
    });

    benchmark("L=3 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 3>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples, 0, nullptr, nullptr);
    });
    std::cout << std::endl;

    if (l_max == 3) {
        free(buffers);
        return;
    }

    //========================================================================//
    compute_sph_prefactors(4, prefactors.data());
    benchmark("L=4 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 4, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=4 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 4, prefactors.data(), xyz.data(), sph.data(), dsph.data(), nullptr, buffers);
    });

    benchmark("L=4 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 4>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples, 0, nullptr, nullptr);
    });

    benchmark("L=4 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 4>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples, 0, nullptr, nullptr);
    });
    std::cout << std::endl;

    if (l_max == 4) {
        free(buffers);
        return;
    }

    //========================================================================//
    compute_sph_prefactors(5, prefactors.data());
    benchmark("L=5 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 5, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=5 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 5, prefactors.data(), xyz.data(), sph.data(), dsph.data(), nullptr, buffers);
    });

    benchmark("L=5 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 5>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples, 0, nullptr, nullptr);
    });

    benchmark("L=5 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 5>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples, 0, nullptr, nullptr);
    });
    std::cout << std::endl;

    if (l_max == 5) {
        free(buffers);
        return;
    }

    //========================================================================//
    compute_sph_prefactors(6, prefactors.data());
    benchmark("L=6 (no h-c) values             ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 6, prefactors.data(), xyz.data(), sph.data(), nullptr, nullptr, buffers);
    });

    benchmark("L=6 (no h-c) values+derivatives ", n_samples, n_tries, [&](){
        compute_generic<DTYPE>(n_samples, 6, prefactors.data(), xyz.data(), sph.data(), dsph.data(), nullptr, buffers);
    });

    benchmark("L=6 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, false, 6>(xyz.data(), sph1.data(), nullptr, nullptr, n_samples, 0, nullptr, nullptr);
    });

    benchmark("L=6 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, false, 6>(xyz.data(), sph1.data(), dsph1.data(), nullptr, n_samples, 0, nullptr, nullptr);
    });
    std::cout << std::endl;

    free(buffers);
}


int main(int argc, char *argv[]) {
    size_t n_samples = 10000;
    size_t n_tries = 1000;
    size_t l_max = 10;

    // parse command line options
    int c;
    while ((c = getopt (argc, argv, "l:s:t:")) != -1) {
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

    std::cout << "Running with n_tries=" << n_tries << ", n_samples=" << n_samples << std::endl;
    std::cout << "\n============= l_max = " << l_max << " ==============" << std::endl;

    std::cout << "****************** SINGLE PRECISION ******************" << std::endl;
    run_timings<float>(l_max, n_tries, n_samples);

    std::cout << "****************** DOUBLE PRECISION ******************" << std::endl;
    run_timings<double>(l_max, n_tries, n_samples);

    return 0;
}
