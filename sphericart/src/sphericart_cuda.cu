#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "cuda_base.hpp"
#include "sphericart_cuda.hpp"

/*host macro that checks for errors in CUDA calls, and prints the file + line
 * and error string if one occurs
 */

using namespace std;
using namespace sphericart::cuda;

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

template <typename T>
SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {
    /*
        This is the constructor of the SphericalHarmonics class. It initizlizes
       buffer space, compute prefactors, and sets the function pointers that are
       used for the actual calls
    */

    this->l_max = (int)l_max;
    this->nprefactors = (int)(l_max + 1) * (l_max + 2);
    this->normalized = normalized;
    this->prefactors_cpu = new T[this->nprefactors];

    // compute prefactors on host first
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);
    // allocate them on device and copy to device
    CUDA_CHECK(cudaMalloc((void **)&this->prefactors_cuda,
                          this->nprefactors * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(this->prefactors_cuda, this->prefactors_cpu,
                          this->nprefactors * sizeof(T),
                          cudaMemcpyHostToDevice));

    // initialise the currently available amount of shared memory.
    this->_current_shared_mem_allocation = adjust_shared_memory(
        sizeof(T), this->l_max, this->CUDA_GRID_DIM_X_, this->CUDA_GRID_DIM_Y_,
        false, false, this->_current_shared_mem_allocation);
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    delete[](this->prefactors_cpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(this->prefactors_cuda));
}

template <typename T>
void SphericalHarmonics<T>::update_cache_and_smem(bool compute_with_gradients,
                                                  bool compute_with_hessian) {

    if (this->cached_compute_with_gradients != compute_with_gradients ||
        this->cached_compute_with_hessian != compute_with_hessian) {

        this->_current_shared_mem_allocation = adjust_shared_memory(
            sizeof(T), this->l_max, this->CUDA_GRID_DIM_X_,
            this->CUDA_GRID_DIM_Y_, compute_with_gradients,
            compute_with_hessian, this->_current_shared_mem_allocation);

        if (this->_current_shared_mem_allocation == -1) {

            std::cerr << "Warning: Failed to update shared memory size, "
                         "re-attempting with  GRID_DIM_Y = 4\n"
                      << std::endl;

            this->CUDA_GRID_DIM_Y_ = 4;
            this->_current_shared_mem_allocation = adjust_shared_memory(
                sizeof(T), this->l_max, this->CUDA_GRID_DIM_X_,
                this->CUDA_GRID_DIM_Y_, compute_with_gradients,
                compute_with_hessian, this->_current_shared_mem_allocation);

            if (this->_current_shared_mem_allocation == -1) {
                throw std::runtime_error(
                    "Insufficient shared memory available to compute "
                    "spherical_harmonics with requested parameters.");
            }
        }

        this->cached_compute_with_gradients = compute_with_gradients;
        this->cached_compute_with_hessian = compute_with_hessian;
    }
}

template <typename T>
void SphericalHarmonics<T>::compute(const T *xyz, const size_t nsamples, T *sph,
                                    void *cuda_stream) {

    if (sph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "sph ptr initialised, instead nullptr found. Initialise "
            "sph with cudaMalloc.");
    }

    this->update_cache_and_smem(false, false);

    switch (this->CUDA_GRID_DIM_Y_) {
    case 16:
        sphericart::cuda::spherical_harmonics<T, 8, 16>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, cuda_stream);
        break;
    case 8:
        sphericart::cuda::spherical_harmonics<T, 8, 8>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, cuda_stream);
        break;
    case 4:
        sphericart::cuda::spherical_harmonics<T, 8, 4>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, cuda_stream);
        break;
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(const T *xyz,
                                                   size_t nsamples, T *sph,
                                                   T *dsph, void *cuda_stream) {
    if (sph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "sph ptr initialised, instead nullptr found. Initialise "
            "sph with cudaMalloc.");
    }

    if (dsph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "dsph != nullptr since compute_with_gradients = true. "
            "initialise dsph with cudaMalloc.");
    }

    this->update_cache_and_smem(true, false);

    switch (this->CUDA_GRID_DIM_Y_) {
    case 16:
        sphericart::cuda::spherical_harmonics_with_gradients<T, 8, 16>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, cuda_stream);
        break;
    case 8:
        sphericart::cuda::spherical_harmonics_with_gradients<T, 8, 8>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, cuda_stream);
        break;
    case 4:
        sphericart::cuda::spherical_harmonics_with_gradients<T, 8, 4>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, cuda_stream);
        break;
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(const T *xyz, size_t nsamples,
                                                  T *sph, T *dsph, T *ddsph,
                                                  void *cuda_stream) {
    if (sph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "sph ptr initialised, instead nullptr found. Initialise "
            "sph with cudaMalloc.");
    }

    if (dsph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "dsph != nullptr since compute_with_gradients = true. "
            "initialise dsph with cudaMalloc.");
    }

    if (ddsph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "ddsph != nullptr since compute_with_hessian = true. "
            "initialise ddsph with cudaMalloc.");
    }

    this->update_cache_and_smem(true, true);

    switch (this->CUDA_GRID_DIM_Y_) {
    case 16:
        sphericart::cuda::spherical_harmonics_with_hessians<T, 8, 16>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, ddsph, cuda_stream);
        break;
    case 8:
        sphericart::cuda::spherical_harmonics_with_hessians<T, 8, 8>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, ddsph, cuda_stream);
        break;
    case 4:
        sphericart::cuda::spherical_harmonics_with_hessians<T, 8, 4>(
            xyz, nsamples, this->prefactors_cuda, this->nprefactors,
            this->l_max, this->normalized, sph, dsph, ddsph, cuda_stream);
        break;
    }
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::cuda::SphericalHarmonics<float>;
template class sphericart::cuda::SphericalHarmonics<double>;
