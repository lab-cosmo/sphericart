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

using namespace sphericart::cuda;

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

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {
    /*
        This is the constructor of the SphericalHarmonics class. It initizlizes
       buffer space, compute prefactors, and sets the function pointers that are
       used for the actual calls
    */
    this->l_max = (int)l_max;
    this->nprefactors = (int)(l_max + 1) * (l_max + 2);
    this->normalized = normalized;
    this->prefactors_cpu = new T[this->nprefactors];

    CUDA_CHECK(cudaGetDeviceCount(&this->device_count));

    // compute prefactors on host first
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);

    if (this->device_count) {
        int current_device;

        CUDA_CHECK(cudaGetDevice(&current_device));

        // allocate prefactorts on every visible device and copy from host
        this->prefactors_cuda = new T*[this->device_count];

        for (int device = 0; device < this->device_count; device++) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(
                cudaMalloc((void**)&this->prefactors_cuda[device], this->nprefactors * sizeof(T))
            );
            CUDA_CHECK(cudaMemcpy(
                this->prefactors_cuda[device],
                this->prefactors_cpu,
                this->nprefactors * sizeof(T),
                cudaMemcpyHostToDevice
            ));
        }

        // initialise the currently available amount of shared memory on all visible devices
        this->_current_shared_mem_allocation = adjust_shared_memory(
            sizeof(T),
            this->l_max,
            this->CUDA_GRID_DIM_X_,
            this->CUDA_GRID_DIM_Y_,
            false,
            false,
            this->_current_shared_mem_allocation
        );

        // set the context back to the current device
        CUDA_CHECK(cudaSetDevice(current_device));
    }
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    if (this->prefactors_cpu != nullptr) {
        delete[] (this->prefactors_cpu);
        this->prefactors_cpu = nullptr;
    }

    if (this->device_count > 0) {

        int current_device;

        CUDA_CHECK(cudaGetDevice(&current_device));

        for (int device = 0; device < this->device_count; device++) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaDeviceSynchronize());
            if (this->prefactors_cuda != nullptr && this->prefactors_cuda[device] != nullptr) {
                CUDA_CHECK(cudaFree(this->prefactors_cuda[device]));
                this->prefactors_cuda[device] = nullptr;
            }
        }
        this->prefactors_cuda = nullptr;

        CUDA_CHECK(cudaSetDevice(current_device));
    }
}

template <typename T>
void SphericalHarmonics<T>::compute(
    const T* xyz,
    const size_t nsamples,
    bool compute_with_gradients,
    bool compute_with_hessian,
    T* sph,
    T* dsph,
    T* ddsph,
    void* cuda_stream
) {
    if (nsamples == 0) {
        // nothing to compute; we return here because some libraries (e.g. torch)
        // seem to use nullptrs for tensors with 0 elements
        return;
    }

    if (sph == nullptr) {
        throw std::runtime_error("sphericart::cuda::SphericalHarmonics::compute expected "
                                 "sph ptr initialised, instead nullptr found. Initialise "
                                 "sph with cudaMalloc.");
    }

    if (compute_with_gradients && dsph == nullptr) {
        throw std::runtime_error("sphericart::cuda::SphericalHarmonics::compute expected "
                                 "dsph != nullptr since compute_with_gradients = true. "
                                 "initialise dsph with cudaMalloc.");
    }

    if (compute_with_hessian && ddsph == nullptr) {
        throw std::runtime_error("sphericart::cuda::SphericalHarmonics::compute expected "
                                 "ddsph != nullptr since compute_with_hessian = true. "
                                 "initialise ddsph with cudaMalloc.");
    }

    if (this->cached_compute_with_gradients != compute_with_gradients ||
        this->cached_compute_with_hessian != compute_with_hessian) {

        this->_current_shared_mem_allocation = adjust_shared_memory(
            sizeof(T),
            this->l_max,
            this->CUDA_GRID_DIM_X_,
            this->CUDA_GRID_DIM_Y_,
            compute_with_gradients,
            compute_with_hessian,
            this->_current_shared_mem_allocation
        );

        if (this->_current_shared_mem_allocation == -1) {

            std::cerr << "Warning: Failed to update shared memory size, "
                         "re-attempting with  GRID_DIM_Y = 4\n"
                      << std::endl;

            this->CUDA_GRID_DIM_Y_ = 4;
            this->_current_shared_mem_allocation = adjust_shared_memory(
                sizeof(T),
                this->l_max,
                this->CUDA_GRID_DIM_X_,
                this->CUDA_GRID_DIM_Y_,
                compute_with_gradients,
                compute_with_hessian,
                this->_current_shared_mem_allocation
            );

            if (this->_current_shared_mem_allocation == -1) {
                throw std::runtime_error("Insufficient shared memory available to compute "
                                         "spherical_harmonics with requested parameters.");
            }
        }

        this->cached_compute_with_gradients = compute_with_gradients;
        this->cached_compute_with_hessian = compute_with_hessian;
    }

    cudaPointerAttributes attributes;

    CUDA_CHECK(cudaPointerGetAttributes(&attributes, xyz));

    int current_device;

    CUDA_CHECK(cudaGetDevice(&current_device));

    if (current_device != attributes.device) {
        CUDA_CHECK(cudaSetDevice(attributes.device));
    }

    sphericart::cuda::spherical_harmonics_cuda_base<T>(
        xyz,
        nsamples,
        this->prefactors_cuda[attributes.device],
        this->nprefactors,
        this->l_max,
        this->normalized,
        this->CUDA_GRID_DIM_X_,
        this->CUDA_GRID_DIM_Y_,
        compute_with_gradients,
        compute_with_hessian,
        sph,
        dsph,
        ddsph,
        cuda_stream
    );

    CUDA_CHECK(cudaSetDevice(current_device));
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::cuda::SphericalHarmonics<float>;
template class sphericart::cuda::SphericalHarmonics<double>;
