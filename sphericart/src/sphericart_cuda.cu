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

using namespace sphericart::cuda;

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
    CUDA_CHECK(cudaMalloc((void **)this->prefactors_cuda,
                          this->nprefactors * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(this->prefactors_cpu, this->prefactors_cuda,
                          this->nprefactors * sizeof(T),
                          cudaMemcpyHostToDevice));
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, frees the prefactors
    delete[] (this->prefactors_cpu);
    CUDA_CHECK(cudaFree(this->prefactors_cuda));
}
template <typename T>
void SphericalHarmonics<T>::compute(const T *xyz, const size_t nsamples,
                                    bool compute_with_gradients,
                                    bool compute_with_hessian,
                                    size_t GRID_DIM_X, size_t GRID_DIM_Y,
                                    T *sph, T *dsph, T *ddsph) {

    if (sph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "sph ptr initialised, instead nullptr found. Initialise "
            "sph with cudaMalloc.");
    }

    if (compute_with_gradients && dsph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "dsph != nullptr since compute_with_gradients = true. "
            "initialise dsph with cudaMalloc.");
    }

    if (compute_with_hessian && ddsph == nullptr) {
        throw std::runtime_error(
            "sphericart::cuda::SphericalHarmonics::compute expected "
            "ddsph != nullptr since compute_with_hessian = true. "
            "initialise ddsph with cudaMalloc.");
    }

    sphericart::cuda::spherical_harmonics_cuda_base<T>(
        xyz, nsamples, this->prefactors_cuda, this->nprefactors, this->l_max,
        this->normalized, GRID_DIM_X, GRID_DIM_Y, compute_with_gradients,
        compute_with_hessian, sph, dsph, ddsph);
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::cuda::SphericalHarmonics<float>;
template class sphericart::cuda::SphericalHarmonics<double>;
