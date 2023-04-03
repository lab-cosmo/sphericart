#include <unistd.h>
#include <sys/time.h>

#include <cmath>
#include <chrono>
#include <iostream>

#include "sphericart.hpp"

#include "../src/templates.hpp"

#define _SPH_TOL 1e-6
#define DTYPE double
using namespace sphericart;

// shorthand for all-past-1 generic sph only
inline void compute_generic(int n_samples, int l_max, DTYPE *prefactors, DTYPE *xyz, DTYPE *sph, DTYPE *dsph) {
    DTYPE * buffers = new DTYPE[100000]; /// TODO remove all this shit
    if (dsph==nullptr) {
        generic_sph<DTYPE, false, false, 1>(xyz, sph, dsph, n_samples, l_max, prefactors, buffers);
    } else {
        generic_sph<DTYPE, true, false, 1>(xyz, sph, dsph, n_samples, l_max, prefactors, buffers);
    }
}

template<typename Fn>
inline void benchmark(std::string context, size_t n_samples, size_t n_tries, Fn function) {
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


int main(int argc, char *argv[]) {
    size_t n_samples = 10000;
    size_t n_tries = 100;
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

    auto prefactors = std::vector<DTYPE>((l_max+1)*(l_max+2), 0.0);
    compute_sph_prefactors(l_max, prefactors.data());

    auto xyz = std::vector<DTYPE>(n_samples*3, 0.0);
    for (size_t i=0; i<n_samples*3; ++i) {
        xyz[i] = (DTYPE)rand()/ (DTYPE) RAND_MAX *2.0-1.0;
    }

    auto sph = std::vector<DTYPE>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph = std::vector<DTYPE>(n_samples*3*(l_max+1)*(l_max+1), 0.0);
    
    benchmark("Call without derivatives", n_samples, n_tries, [&](){
        compute_generic(n_samples, l_max, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("Call with derivatives", n_samples, n_tries, [&](){
        compute_generic(n_samples, l_max, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });
    std::cout << std::endl;

    auto sxyz = std::vector<DTYPE>(3, 0.0);
    auto ssph = std::vector<DTYPE>((l_max+1)*(l_max+1), 0.0);
    auto sdsph = std::vector<DTYPE>(3*(l_max+1)*(l_max+1), 0.0);
    auto sph1 = std::vector<DTYPE>(n_samples*(l_max+1)*(l_max+1), 0.0);
    auto dsph1 = std::vector<DTYPE>(n_samples*3*(l_max+1)*(l_max+1), 0.0);
    
    {
    SphericalHarmonics<DTYPE> SH(l_max, false);
    sxyz[0] = xyz[0]; sxyz[1] = xyz[1]; sxyz[2] = xyz[2]; 

    benchmark("Sample without derivatives", n_samples, n_tries, [&](){
        SH.compute(sxyz, ssph);
    });

    benchmark("Sample with derivatives", n_samples, n_tries, [&](){
        SH.compute(sxyz, ssph, sdsph);
    });

    std::cout << std::endl;
    
    benchmark("Call without derivatives (hybrid)", n_samples, n_tries, [&](){
        SH.compute(xyz, sph1);
    });

    benchmark("Call with derivatives (hybrid)", n_samples, n_tries, [&](){
        SH.compute(xyz, sph1, dsph1);
    });
    }

    std::cout << std::endl;
    
    int size3 = 3*(l_max+1)*(l_max+1);  // Size of the third dimension in derivative arrays (or second in normal sph arrays).
    int size2 = (l_max+1)*(l_max+1);  // Size of the second+third dimensions in derivative arrays
    for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
        for (size_t l=0; l<(l_max+1); l++) {
            for (int m=-static_cast<int>(l); m<=static_cast<int>(l); m++) {
                if (fabs(sph[size2*i_sample+l*l+l+m]/sph1[size2*i_sample+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("SPH: %e, %e\n", sph[size2*i_sample+l*l+l+m], sph1[size2*i_sample+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*0+l*l+l+m]/dsph1[size3*i_sample+size2*0+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DxSPH: %e, %e\n", dsph[size3*i_sample+size2*0+l*l+l+m], dsph1[size3*i_sample+size2*0+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*1+l*l+l+m]/dsph1[size3*i_sample+size2*1+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DySPH: %e, %e\n", dsph[size3*i_sample+size2*1+l*l+l+m],dsph1[size3*i_sample+size2*1+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*2+l*l+l+m]/dsph1[size3*i_sample+size2*2+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DzSPH: %e, %e\n", dsph[size3*i_sample+size2*2+l*l+l+m], dsph1[size3*i_sample+size2*2+l*l+l+m]);
                }
            }
        }
    }

    {
    SphericalHarmonics<DTYPE> SH(l_max, true);
    benchmark("Call without derivatives (hybrid, normalized)", n_samples, n_tries, [&](){
        SH.compute(xyz, sph1);
    });

    benchmark("Call with derivatives (hybrid, normalized)", n_samples, n_tries, [&](){
        SH.compute(xyz, sph1, dsph1);
    });
    }
    std::cout << std::endl;

    std::cout << "================ Low-l timings ===========" << std::endl;

    compute_sph_prefactors(1, prefactors.data());
    benchmark("L=1 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 1, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=1 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 1, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=1 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 1>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=1 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 1>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    //========================================================================//
    compute_sph_prefactors(2, prefactors.data());
    benchmark("L=2 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 2, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=2 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 2, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=2 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 2>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=2 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 2>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    //========================================================================//
    compute_sph_prefactors(3, prefactors.data());
    benchmark("L=3 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 3, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=3 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 3, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=3 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 3>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=3 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 3>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    //========================================================================//
    compute_sph_prefactors(4, prefactors.data());
    benchmark("L=4 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 4, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=4 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 4, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=4 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 4>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=4 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 4>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    //========================================================================//
    compute_sph_prefactors(5, prefactors.data());
    benchmark("L=5 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 5, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=5 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 5, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=5 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 5>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=5 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 5>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    //========================================================================//
    compute_sph_prefactors(6, prefactors.data());
    benchmark("L=6 values                      ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 6, prefactors.data(), xyz.data(), sph.data(), nullptr);
    });

    benchmark("L=6 values+derivatives          ", n_samples, n_tries, [&](){
        compute_generic(n_samples, 6, prefactors.data(), xyz.data(), sph.data(), dsph.data());
    });

    benchmark("L=6 hardcoded values            ", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, false, false, 6>(xyz.data(), sph1.data(), nullptr,n_samples,  0, nullptr, nullptr);
    });

    benchmark("L=6 hardcoded values+derivatives", n_samples, n_tries, [&](){
        hardcoded_sph<DTYPE, true, false, 6>(xyz.data(), sph1.data(), dsph1.data(),n_samples,  0, nullptr, nullptr);
    });
    std::cout << std::endl;

    l_max=6;
    size3 = 3*(l_max+1)*(l_max+1);  // Size of the third dimension in derivative arrays (or second in normal sph arrays).
    size2 = (l_max+1)*(l_max+1);  // Size of the second+third dimensions in derivative arrays
    for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
        for (size_t l=0; l<(l_max+1); l++) {
            for (int m=-static_cast<int>(l); m<=static_cast<int>(l); m++) {
                if (fabs(sph[size2*i_sample+l*l+l+m]/sph1[size2*i_sample+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("SPH: %e, %e\n", sph[size2*i_sample+l*l+l+m], sph1[size2*i_sample+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*0+l*l+l+m]/dsph1[size3*i_sample+size2*0+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DxSPH: %e, %e\n", dsph[size3*i_sample+size2*0+l*l+l+m], dsph1[size3*i_sample+size2*0+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*1+l*l+l+m]/dsph1[size3*i_sample+size2*1+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DySPH: %e, %e\n", dsph[size3*i_sample+size2*1+l*l+l+m],dsph1[size3*i_sample+size2*1+l*l+l+m]);
                }
                if (fabs(dsph[size3*i_sample+size2*2+l*l+l+m]/dsph1[size3*i_sample+size2*2+l*l+l+m]-1)>_SPH_TOL) {
                    printf("Problem detected at i_sample = %zu, L = %zu, m = %d \n", i_sample, l, m);
                    printf("DzSPH: %e, %e\n", dsph[size3*i_sample+size2*2+l*l+l+m], dsph1[size3*i_sample+size2*2+l*l+l+m]);
                }
            }
        }
    }

    return 0;
}
