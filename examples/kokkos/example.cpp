/** @file example.cu
 *  @brief Usage example for the CUDA C++ API
 */

// #include "sphericart_cuda.hpp"
// #include <cmath>
// #include <cstdio>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

#include <Kokkos_Core.hpp>
#include <sycl/sycl.hpp>

#include "sphericart_sycl.hpp"
#include "sphericart.hpp"

#include <iostream>
#include <iomanip>

#include <cmath>
#include <cstdio>
#include <vector>

#ifndef DTYPE
#define DTYPE double
#endif

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

        /* ===== set up the calculation ===== */

        // hard-coded parameters for the example
        size_t n_samples = 1000;
        size_t l_max = 10;

        // initializes samples
        auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
        for (size_t i = 0; i < n_samples * 3; ++i) {
            xyz[i] = (DTYPE)rand() / (DTYPE)RAND_MAX * 2.0 - 1.0;
        }

        // to avoid unnecessary allocations, calculators can use pre-allocated
        // memory, one also can provide uninitialized vectors that will be
        // automatically reshaped
        auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
        auto dsph = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
        auto ddsph = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

        // the class is templated, so one can also use 32-bit DTYPE operations
        auto xyz_f = std::vector<DTYPE>(n_samples * 3, 0.0);
        for (size_t i = 0; i < n_samples * 3; ++i) {
            xyz_f[i] = (DTYPE)xyz[i];
        }
        auto sph_f = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
        auto dsph_f = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
        auto ddsph_f = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

        using ExecSpace = Kokkos::SYCL;
        // GPU device views - all allocated in GPU memory
        // Create Kokkos Views for matrices on GPU (using DTYPE as example)
        using DataType = DTYPE;
        using DeviceSpace = Kokkos::Experimental::SYCLDeviceUSMSpace;
        using HostSpace = Kokkos::HostSpace;

        /* ===== API calls ===== */

        // internal buffers and numerical factors are initalized at construction
        // sphericart::cuda::SphericalHarmonics<DTYPE> calculator_cuda(l_max);

        // GPU device views - all allocated in GPU memory
        Kokkos::View<DataType*, DeviceSpace> xyz_device("xyz_device", xyz_f.size());
        Kokkos::View<DataType*, DeviceSpace> sph_device("sph_device", sph_f.size());
        Kokkos::View<DataType*, DeviceSpace> dsph_device("dsph_device", dsph_f.size());
        Kokkos::View<DataType*, DeviceSpace> ddsph_device("ddsph_device", ddsph_f.size());
        // Host views for initialization and verification
        Kokkos::View<DataType*, HostSpace> xyz_host("xyz_host", xyz_f.size());
        Kokkos::View<DataType*, HostSpace> sph_host("sph_host", sph_f.size());
        Kokkos::View<DataType*, HostSpace> dsph_host("dsph_host", dsph_f.size());
        Kokkos::View<DataType*, HostSpace> ddsph_host("ddsph_host", ddsph_f.size());

        Kokkos::fence(); // Ensure any previous operations are done

        std::copy(xyz_f.begin(), xyz_f.end(), xyz_host.data());
        Kokkos::deep_copy(xyz_device, xyz_host);

        DTYPE* xyz_acc = xyz_device.data();
        DTYPE* sph_acc = sph_device.data();
        DTYPE* dsph_acc = dsph_device.data();
        DTYPE* ddsph_acc = ddsph_device.data();

        // GPU Calculator
        sphericart::sycl::SphericalHarmonics<DTYPE> calculator_sycl(l_max);
        // CPU Calculator
        auto calculator = sphericart::SphericalHarmonics<DTYPE>(l_max);

        calculator_sycl.compute(xyz_acc, n_samples, sph_acc); // no gradients GPU
        calculator.compute(xyz, sph);                         // no gradients CPU

        Kokkos::deep_copy(sph_host, sph_device);

        // calculation examples

        //    int size2 = (l_max + 1) * (l_max + 1); // Size of the second+third dimensions in
        //    derivative arrays
        // for (size_t i_sample = 0; i_sample < 10; i_sample++) {
        //    for (size_t l = 0; l < (l_max + 1); l++) {
        //        for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {

        //                printf(
        //                    "SPH: %e , %e\n",
        //                    sph_host[size2 * i_sample + l * l + l + m],
        //                    sph[size2 * i_sample + l * l + l + m]
        //                );
        //        }
        //    }
        // }

        /* ===== check results ===== */

        DTYPE sph_error = 0.0, sph_norm = 0.0;
        for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
            sph_error += (sph_host[i] - sph[i]) * (sph_host[i] - sph[i]);
            sph_norm += sph[i] * sph[i];
        }
        printf("GPU XPU vs CPU relative error: %12.8e\n", sqrt(sph_error / sph_norm));
    }
    Kokkos::finalize();
    return 0;
}
