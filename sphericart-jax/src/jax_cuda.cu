// This file defines the Python interface to the XLA custom calls.
// It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>



#include "sphericart/jax_cuda.hpp"
#include "sphericart/pybind11_kernel_helpers.h"

#include "sphericart_cuda.hpp"

#include <iostream>

using namespace std;


namespace sphericart_jax {

namespace cuda {
    
template <typename T>
using CacheMapCUDA =
    std::map<std::tuple<size_t, bool>,
             std::unique_ptr<sphericart::cuda::SphericalHarmonics<T>>>;

template <typename T>
std::unique_ptr<sphericart::cuda::SphericalHarmonics<T>> &
_get_or_create_sph_cuda(CacheMapCUDA<T> &sph_cache, std::mutex &cache_mutex,
                        size_t l_max, bool normalized) {
    // Check if instance exists in cache, if not create and store it
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto key = std::make_tuple(l_max, normalized);
    auto it = sph_cache.find(key);
    if (it == sph_cache.end()) {
        it = sph_cache
                 .insert(
                     {key,
                      std::make_unique<sphericart::cuda::SphericalHarmonics<T>>(
                          l_max, normalized)})
                 .first;
    }
    return it->second;
}

template <typename T>
inline void cuda_sph(cudaStream_t stream, void **in, const char *opaque,
                     std::size_t opaque_len) {
    // Parse the inputs
    std::cout << "STILL OK 0" << std::endl;
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    // const size_t l_max = *reinterpret_cast<const int *>(in[1]);
    // const bool normalized = *reinterpret_cast<const bool *>(in[2]);
    // const size_t n_samples = *reinterpret_cast<const int *>(in[3]);
    // size_t xyz_length{n_samples * 3};
    // size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    //  The output is stored as a single pointer since there is only one output
    // T *sph = reinterpret_cast<T *>(out);
    T *sph = reinterpret_cast<T *>(in[1]);
    // Static map to cache instances based on parameters
    static CacheMapCUDA<T> sph_cache;
    static std::mutex cache_mutex;

    const SphDescriptor &d =
        *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;
    const bool normalize = d.normalize;

    cout << "n_samples: " << n_samples << endl;
    cout << "lmax: " << lmax << endl;
    cout << "normalize: " << normalize << endl;

    auto &calculator =
        _get_or_create_sph_cuda(sph_cache, cache_mutex, lmax, normalize);
    std::cout << "STILL OK 1" << std::endl;

    cout << "xyz_pointer: " << xyz << endl;
    cout << "sph_pointer: " << sph << endl;

    calculator->compute(xyz, n_samples, false, false, sph, nullptr, nullptr);
    std::cout << "STILL OK 2" << std::endl;
}

void apply_cuda_sph_f32(cudaStream_t stream, void **in,
                                              const char *opaque,
                                              std::size_t opaque_len) {
    cuda_sph<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_f64(cudaStream_t stream, void **in,
                                              const char *opaque,
                                              std::size_t opaque_len) {
    cuda_sph<double>(stream, in, opaque, opaque_len);
}

template <typename T>
inline void cuda_sph_with_gradients(cudaStream_t stream, void **in,
                                    const char *opaque,
                                    std::size_t opaque_len) {
    std::cout << "STILL OK 0 grad" << std::endl;
    // Parse the inputs
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    T *sph = reinterpret_cast<T *>(in[1]);
    T *dsph = reinterpret_cast<T *>(in[2]);

    // Static map to cache instances based on parameters
    static CacheMapCUDA<T> sph_cache;
    static std::mutex cache_mutex;

    const SphDescriptor &d =
        *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;
    const bool normalize = d.normalize;

    auto &calculator =
        _get_or_create_sph_cuda(sph_cache, cache_mutex, lmax, normalize);
    std::cout << "STILL OK 1 grAD" << std::endl;
    calculator->compute(xyz, n_samples, true, false, sph, dsph, nullptr);
    std::cout << "STILL OK 2 GRAD" << std::endl;
}

void apply_cuda_sph_with_gradients_f32(
    cudaStream_t stream, void **in, const char *opaque,
    std::size_t opaque_len) {
    cuda_sph_with_gradients<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_with_gradients_f64(
    cudaStream_t stream, void **in, const char *opaque,
    std::size_t opaque_len) {
    cuda_sph_with_gradients<double>(stream, in, opaque, opaque_len);
}

template <typename T>
inline void cuda_sph_with_hessians(cudaStream_t stream, void **in,
                                   const char *opaque, std::size_t opaque_len) {
    std::cout << "STILL OK 0 hess" << std::endl;
    // Parse the inputs
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    T *sph = reinterpret_cast<T *>(in[1]);
    T *dsph = reinterpret_cast<T *>(in[2]);
    T *ddsph = reinterpret_cast<T *>(in[3]);

    // Static map to cache instances based on parameters
    static CacheMapCUDA<T> sph_cache;
    static std::mutex cache_mutex;

    const SphDescriptor &d =
        *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;
    const bool normalize = d.normalize;

    auto &calculator =
        _get_or_create_sph_cuda(sph_cache, cache_mutex, lmax, normalize);
    std::cout << "STILL OK 1 hes" << std::endl;
    calculator->compute(xyz, n_samples, true, true, sph, dsph, ddsph);
    std::cout << "STILL OK 2 hess" << std::endl;
}

void apply_cuda_sph_with_hessians_f32(
    cudaStream_t stream, void **in, const char *opaque,
    std::size_t opaque_len) {
    cuda_sph_with_hessians<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_with_hessians_f64(
    cudaStream_t stream, void **in, const char *opaque,
    std::size_t opaque_len) {
    cuda_sph_with_hessians<double>(stream, in, opaque, opaque_len);
}
} // namespace cuda
} // namespace sphericart_jax
