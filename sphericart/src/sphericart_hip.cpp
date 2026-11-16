#include <stdexcept>

#include <hip/hip_runtime.h>

#define _SPHERICART_INTERNAL_IMPLEMENTATION

#include "hip_base.hpp"
#include "sphericart_hip.hpp"
#include "templates.hpp"
#include "hiprtc_lite.hpp"

using namespace sphericart::hip;

/*
    This code checks whether the HIP runtime and the runtime compilation
    library we need are available on the host. The HIP runtime is a link-time
    dependency of this backend, while libhiprtc is resolved dynamically at
    runtime (the mechanism ported from symmetrix-xl's HIP backend).
*/

static void checkHip() {
    int device_count = 0;
    hipError_t status = hipGetDeviceCount(&device_count);
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string("Failed to initialize the HIP runtime: ") + hipGetErrorString(status)
        );
    }
    if (device_count < 1) {
        throw std::runtime_error(
            "No HIP devices are available in a usable state. Check that the "
            "AMD GPU is visible to the ROCm runtime."
        );
    }

    auto hiprtc = sphericart::hip::hiprtc_information();
    if (!hiprtc.available) {
        throw std::runtime_error(
            "Failed to load libhiprtc.so, which is required to compile "
            "sphericart's HIP kernels at runtime. Set SPHERICART_HIPRTC_LIBRARY "
            "to the full path of libhiprtc.so, or append the directory "
            "containing this library (e.g. /opt/rocm/lib) to your "
            "$LD_LIBRARY_PATH environment variable. (" +
            hiprtc.reason + ")"
        );
    }
}

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t l_max) {
    /*
        This is the constructor of the SphericalHarmonics class. It initializes
       buffer space, compute prefactors, and sets the function pointers that are
       used for the actual calls
    */

    checkHip();

    this->l_max = (int)l_max;
    this->nprefactors = (int)(l_max + 1) * (l_max + 2);
    this->normalized = true; // SphericalHarmonics class
    this->prefactors_cpu = new T[this->nprefactors];

    hipError_t status = hipGetDeviceCount(&this->device_count);
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string("Failed to get the number of HIP devices: ") + hipGetErrorString(status)
        );
    }

    // compute prefactors on host
    compute_sph_prefactors<T>((int)l_max, this->prefactors_cpu);
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    if (this->prefactors_cpu != nullptr) {
        delete[] (this->prefactors_cpu);
        this->prefactors_cpu = nullptr;
    }

    /* Leave OS to cleanup due to static (de-)initialisation problem with sphericart-jax.
    if (this->prefactors_hip) {
        hipFree(this->prefactors_hip);
    }
    */
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
    void* hip_stream,
    bool synchronize
) {
    if (n_samples == 0) {
        // nothing to compute; we return here because some libraries (e.g. torch)
        // seem to use nullptrs for tensors with 0 elements
        return;
    }

    if (sph == nullptr) {
        throw std::runtime_error(
            "sphericart::hip::SphericalHarmonics::compute expected "
            "sph ptr initialised, instead nullptr found. Initialise "
            "sph with hipMalloc."
        );
    }

    if (compute_with_gradients && dsph == nullptr) {
        throw std::runtime_error(
            "sphericart::hip::SphericalHarmonics::compute expected "
            "dsph != nullptr since compute_with_gradients = true. "
            "initialise dsph with hipMalloc."
        );
    }

    if (compute_with_hessian && ddsph == nullptr) {
        throw std::runtime_error(
            "sphericart::hip::SphericalHarmonics::compute expected "
            "ddsph != nullptr since compute_with_hessian = true. "
            "initialise ddsph with hipMalloc."
        );
    }

    hipPointerAttribute_t attributes;

    hipError_t status = hipPointerGetAttributes(&attributes, xyz);
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(
                "sphericart::hip::SphericalHarmonics::compute could not "
                "resolve the device of the input pointer: "
            ) +
            hipGetErrorString(status)
        );
    }

    int current_device;

    status = hipGetDevice(&current_device);
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("hipGetDevice failed: ") + hipGetErrorString(status));
    }

    if (this->bound_device >= 0 && this->bound_device != attributes.device) {
        throw std::runtime_error(
            "sphericart HIP calculator device migration is unsupported; the calculator is bound "
            "to its first device"
        );
    }
    if (current_device != attributes.device) {
        status = hipSetDevice(attributes.device);
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string("hipSetDevice failed: ") + hipGetErrorString(status)
            );
        }
    }

    if (!this->prefactors_hip) {
        status = hipMalloc((void**)&this->prefactors_hip, this->nprefactors * sizeof(T));
        if (status != hipSuccess) {
            throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(status));
        }
        this->bound_device = attributes.device;

        status = hipMemcpy(
            this->prefactors_hip,
            this->prefactors_cpu,
            this->nprefactors * sizeof(T),
            hipMemcpyHostToDevice
        );
        if (status != hipSuccess) {
            throw std::runtime_error(std::string("hipMemcpy failed: ") + hipGetErrorString(status));
        }
    }

    sphericart::hip::spherical_harmonics_hip_base<T>(
        xyz,
        n_samples,
        this->prefactors_hip,
        this->nprefactors,
        this->l_max,
        this->normalized,
        this->HIP_GRID_DIM_X_,
        this->HIP_GRID_DIM_Y_,
        compute_with_gradients,
        compute_with_hessian,
        sph,
        dsph,
        ddsph,
        hip_stream,
        synchronize
    );

    if (current_device != attributes.device) {
        status = hipSetDevice(current_device);
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string("hipSetDevice failed: ") + hipGetErrorString(status)
            );
        }
    }
}

template <typename T>
void SphericalHarmonics<T>::compute(const T* xyz, const size_t n_samples, T* sph, void* hip_stream) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, false, false, sph, nullptr, nullptr, hip_stream, true
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, void* hip_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, false, sph, dsph, nullptr, hip_stream, true
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients_async(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, void* hip_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, false, sph, dsph, nullptr, hip_stream, false
    );
}

template <typename T> void SphericalHarmonics<T>::release_device_resources() {
    if (this->prefactors_hip == nullptr) {
        return;
    }

    if (this->bound_device < 0) {
        throw std::runtime_error("sphericart HIP calculator has no bound device");
    }

    int current_device;
    hipError_t status = hipGetDevice(&current_device);
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("hipGetDevice failed: ") + hipGetErrorString(status));
    }

    const bool restore_device = current_device != this->bound_device;
    if (restore_device) {
        status = hipSetDevice(this->bound_device);
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string("hipSetDevice failed: ") + hipGetErrorString(status)
            );
        }
    }

    const hipError_t free_status = hipFree(this->prefactors_hip);
    if (free_status == hipSuccess) {
        this->prefactors_hip = nullptr;
    }
    hipError_t restore_status = hipSuccess;
    if (restore_device) {
        restore_status = hipSetDevice(current_device);
    }
    if (free_status != hipSuccess) {
        throw std::runtime_error(std::string("hipFree failed: ") + hipGetErrorString(free_status));
    }
    if (restore_status != hipSuccess) {
        throw std::runtime_error(
            std::string("hipSetDevice failed: ") + hipGetErrorString(restore_status)
        );
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    const T* xyz, const size_t n_samples, T* sph, T* dsph, T* ddsph, void* hip_stream
) {
    SphericalHarmonics<T>::compute_internal(
        xyz, n_samples, true, true, sph, dsph, ddsph, hip_stream, true
    );
}

template <typename T>
SolidHarmonics<T>::SolidHarmonics(size_t l_max) : SphericalHarmonics<T>(l_max) {
    this->normalized = false; // SolidHarmonics class
}

// instantiates the SphericalHarmonics and SolidHarmonics classes
// for basic floating point types
template class sphericart::hip::SphericalHarmonics<float>;
template class sphericart::hip::SphericalHarmonics<double>;
template class sphericart::hip::SolidHarmonics<float>;
template class sphericart::hip::SolidHarmonics<double>;
