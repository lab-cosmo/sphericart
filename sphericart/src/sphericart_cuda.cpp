#include <iostream>
#include <stdexcept>

#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include "dynamic_cuda.hpp"
#include "sphericart.hpp"
#include "cuda_base.hpp"
#include "sphericart_cuda.hpp"

using namespace sphericart::cuda;

/*
    This code checks whether the cuda libraries we'd need to dynamically load are available on the host
*/

void checkCuda() {
    if (!CUDADriver::instance().loaded()) {
        throw std::runtime_error(
            "Failed to load libcuda.so. Try running \"find /usr -name libcuda.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!CUDART::instance().loaded()) {
        throw std::runtime_error(
            "Failed to load libcudart.so. Try running \"find /usr -name libcudart.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!NVRTC::instance().loaded()) {
        throw std::runtime_error(
            "Failed to load libnvrtc.so. Try running \"find /usr -name libnvrtc.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable."
        );
    }
}

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t l_max) {
    /*
        This is the constructor of the SphericalHarmonics class. It initizlizes
       buffer space, compute prefactors, and sets the function pointers that are
       used for the actual calls
    */

    checkCuda();

    this->l_max = (int)l_max;
    this->nprefactors = (int)(l_max + 1) * (l_max + 2);
    this->normalized = true; // SphericalHarmonics class
    this->prefactors_cpu = new T[this->nprefactors];

    CUDART_SAFE_CALL(CUDART::instance().cudaGetDeviceCount(&this->device_count));

    // compute prefactors on host
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    if (this->prefactors_cpu != nullptr) {
        delete[](this->prefactors_cpu);
        this->prefactors_cpu = nullptr;
    }

    if (this->prefactors_cuda) {
        CUDART_SAFE_CALL(CUDART::instance().cudaFree(this->prefactors_cuda));
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_internal(
    const T* xyz,
    const size_t n_samples,
    bool compute_with_gradients,
    bool compute_with_hessian,
    T* sph,
    T* dsph,
    T* ddsph,
    void* cuda_stream
) {
    if (n_samples == 0) {
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

    cudaPointerAttributes attributes;

    CUDART_SAFE_CALL(CUDART::instance().cudaPointerGetAttributes(&attributes, xyz));

    int current_device;

    CUDART_SAFE_CALL(CUDART::instance().cudaGetDevice(&current_device));

    if (current_device != attributes.device) {
        CUDART_SAFE_CALL(CUDART::instance().cudaSetDevice(attributes.device));
    }

    if (!this->prefactors_cuda) {
        CUDART_SAFE_CALL(CUDART::instance().cudaMalloc(
            (void**)&this->prefactors_cuda, this->nprefactors * sizeof(T)
        ));

        CUDART_SAFE_CALL(CUDART::instance().cudaMemcpy(
            this->prefactors_cuda,
            this->prefactors_cpu,
            this->nprefactors * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }

    sphericart::cuda::spherical_harmonics_cuda_base<T>(
        xyz,
        n_samples,
        this->prefactors_cuda,
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

    if (current_device != attributes.device) {
        CUDART_SAFE_CALL(CUDART::instance().cudaSetDevice(current_device));
    }
}
template <typename T>
void SphericalHarmonics<T>::compute(const T* xyz, const size_t n_samples, T* sph, void* cuda_stream) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, false, false, sph, nullptr, nullptr, cuda_stream
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, void* cuda_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, false, sph, dsph, nullptr, cuda_stream
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, T* ddsph, void* cuda_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, true, sph, dsph, ddsph, cuda_stream
    );
}

template <typename T>
SolidHarmonics<T>::SolidHarmonics(size_t l_max) : SphericalHarmonics<T>(l_max) {
    this->normalized = false; // SolidHarmonics class
}

// instantiates the SphericalHarmonics and SolidHarmonics classes
// for basic floating point types
template class sphericart::cuda::SphericalHarmonics<float>;
template class sphericart::cuda::SphericalHarmonics<double>;
template class sphericart::cuda::SolidHarmonics<float>;
template class sphericart::cuda::SolidHarmonics<double>;
