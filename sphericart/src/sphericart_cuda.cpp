#include <iostream>
#include <stdexcept>

#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include "dynamic_cuda.hpp"
#include "sphericart.hpp"
#include "cuda_base.hpp"
#include "sphericart_cuda.hpp"

/*host macro that checks for errors in CUDA calls, and prints the file + line
 * and error string if one occurs
 */

using namespace sphericart::cuda;

struct CudaError {
    bool success;
    std::string errorMessage;
};

CudaError isCudaAvailable() {
    void* cudaHandle = dlopen("libcuda.so", RTLD_NOW);
    void* nvrtcHandle = dlopen("libnvrtc.so", RTLD_NOW);
    void* cudartHandle = dlopen("libcudart.so", RTLD_NOW);

    CudaError result;
    result.success = true;

    if (!cudaHandle) {
        result.success = false;
        result.errorMessage =
            "Failed to load libcuda.so. Try running \"find /usr -name libcuda.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable.";
    } else if (!cudartHandle) {
        result.success = false;
        result.errorMessage =
            "Failed to load libcudart.so. Try running \"find /usr -name libcudart.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable.";
    } else if (!nvrtcHandle) {
        result.success = false;
        result.errorMessage =
            "Failed to load libnvrtc.so. Try running \"find /usr -name libnvrtc.so\" and "
            "appending the directory to your $LD_LIBRARY_PATH environment variable.";
    }

    // Close libraries if they were opened
    if (cudaHandle)
        dlclose(cudaHandle);
    if (cudartHandle)
        dlclose(cudartHandle);
    if (nvrtcHandle)
        dlclose(nvrtcHandle);

    return result;
}

void checkCuda() {
    CudaError cudaCheck = isCudaAvailable();
    if (!cudaCheck.success) {
        throw std::runtime_error(cudaCheck.errorMessage);
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

    CUDART_SAFE_CALL(DynamicCUDA::instance().cudaGetDeviceCount(&this->device_count));

    // compute prefactors on host first
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);

    if (this->device_count) {
        int current_device;

        CUDART_SAFE_CALL(DynamicCUDA::instance().cudaGetDevice(&current_device));

        // allocate prefactorts on every visible device and copy from host
        this->prefactors_cuda = new T*[this->device_count];

        for (int device = 0; device < this->device_count; device++) {
            CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(device));
            CUDART_SAFE_CALL(DynamicCUDA::instance().cudaMalloc(
                (void**)&this->prefactors_cuda[device], this->nprefactors * sizeof(T)
            ));
            CUDART_SAFE_CALL(DynamicCUDA::instance().cudaMemcpy(
                this->prefactors_cuda[device],
                this->prefactors_cpu,
                this->nprefactors * sizeof(T),
                cudaMemcpyHostToDevice
            ));
        }

        // set the context back to the current device
        CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(current_device));
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

        CUDART_SAFE_CALL(DynamicCUDA::instance().cudaGetDevice(&current_device));

        for (int device = 0; device < this->device_count; device++) {
            CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(device));
            CUDART_SAFE_CALL(DynamicCUDA::instance().cudaDeviceSynchronize());
            if (this->prefactors_cuda != nullptr && this->prefactors_cuda[device] != nullptr) {
                CUDART_SAFE_CALL(DynamicCUDA::instance().cudaFree(this->prefactors_cuda[device]));
                this->prefactors_cuda[device] = nullptr;
            }
        }
        this->prefactors_cuda = nullptr;

        CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(current_device));
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_internal(
    T* xyz,
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

    CUDART_SAFE_CALL(DynamicCUDA::instance().cudaPointerGetAttributes(&attributes, xyz));

    int current_device;

    CUDART_SAFE_CALL(DynamicCUDA::instance().cudaGetDevice(&current_device));

    if (current_device != attributes.device) {
        CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(attributes.device));
    }

    sphericart::cuda::spherical_harmonics_cuda_base<T>(
        xyz,
        n_samples,
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

    CUDART_SAFE_CALL(DynamicCUDA::instance().cudaSetDevice(current_device));
}
template <typename T>
void SphericalHarmonics<T>::compute(T* xyz, const size_t n_samples, T* sph, void* cuda_stream) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, false, false, sph, nullptr, nullptr, cuda_stream
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    T* xyz, const size_t n_samples, T* sph, T* dsph, void* cuda_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, false, sph, dsph, nullptr, cuda_stream
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    T* xyz, const size_t n_samples, T* sph, T* dsph, T* ddsph, void* cuda_stream
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
