// This file implements the CUDA interface between JAX and the CUDA C++
// implementation of sphericart.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include "sphericart/sphericart_jax_cuda.hpp"
#include "sphericart/pybind11_kernel_helpers.hpp"

#include "sphericart_cuda.hpp"

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
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    T *sph = reinterpret_cast<T *>(in[1]);

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

    calculator->compute(xyz, n_samples, false, false, sph, nullptr, nullptr,
                        reinterpret_cast<void *>(stream));
}

void apply_cuda_sph_f32(cudaStream_t stream, void **in, const char *opaque,
                        std::size_t opaque_len) {
    cuda_sph<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_f64(cudaStream_t stream, void **in, const char *opaque,
                        std::size_t opaque_len) {
    cuda_sph<double>(stream, in, opaque, opaque_len);
}

template <typename T>
inline void cuda_sph_with_gradients(cudaStream_t stream, void **in,
                                    const char *opaque,
                                    std::size_t opaque_len) {
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
    calculator->compute(xyz, n_samples, true, false, sph, dsph, nullptr,
                        reinterpret_cast<void *>(stream));
}

void apply_cuda_sph_with_gradients_f32(cudaStream_t stream, void **in,
                                       const char *opaque,
                                       std::size_t opaque_len) {
    cuda_sph_with_gradients<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_with_gradients_f64(cudaStream_t stream, void **in,
                                       const char *opaque,
                                       std::size_t opaque_len) {
    cuda_sph_with_gradients<double>(stream, in, opaque, opaque_len);
}

template <typename T>
inline void cuda_sph_with_hessians(cudaStream_t stream, void **in,
                                   const char *opaque, std::size_t opaque_len) {
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
    calculator->compute(xyz, n_samples, true, true, sph, dsph, ddsph,
                        reinterpret_cast<void *>(stream));
}

void apply_cuda_sph_with_hessians_f32(cudaStream_t stream, void **in,
                                      const char *opaque,
                                      std::size_t opaque_len) {
    cuda_sph_with_hessians<float>(stream, in, opaque, opaque_len);
}

void apply_cuda_sph_with_hessians_f64(cudaStream_t stream, void **in,
                                      const char *opaque,
                                      std::size_t opaque_len) {
    cuda_sph_with_hessians<double>(stream, in, opaque, opaque_len);
}
} // namespace cuda
} // namespace sphericart_jax
