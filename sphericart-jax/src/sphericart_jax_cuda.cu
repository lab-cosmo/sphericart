// This file defines the Python interface to the XLA custom calls on CUDA
// devices. It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "sphericart_cuda.hpp"
#include "sphericart/pybind11_kernel_helpers.hpp"
#include "sphericart/sphericart_jax_cuda.hpp"

namespace sphericart_jax {
namespace cuda {

template <template <typename> class C, typename T>
using CacheMapCUDA = std::map<size_t, std::unique_ptr<C<T>>>;

template <template <typename> class C, typename T>
std::unique_ptr<C<T>>& _get_or_create_sph_cuda(size_t l_max) {
    // Static map to cache instances based on parameters
    static CacheMapCUDA<C, T> sph_cache;
    static std::mutex cache_mutex;

    // Check if instance exists in cache, if not create and store it
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = sph_cache.find(l_max);
    if (it == sph_cache.end()) {
        it = sph_cache.insert({l_max, std::make_unique<C<T>>(l_max)}).first;
    }
    return it->second;
}

template <template <typename> class C, typename T>
inline void cuda_sph(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    T* sph = reinterpret_cast<T*>(in[1]);

    const SphDescriptor& d = *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;

    auto& calculator = _get_or_create_sph_cuda<C, T>(lmax);

    calculator->compute(xyz, n_samples, sph, reinterpret_cast<void*>(stream));
}

template <template <typename> class C, typename T>
inline void cuda_sph_with_gradients(
    cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len
) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    T* sph = reinterpret_cast<T*>(in[1]);
    T* dsph = reinterpret_cast<T*>(in[2]);

    const SphDescriptor& d = *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;

    auto& calculator = _get_or_create_sph_cuda<C, T>(lmax);
    calculator->compute_with_gradients(xyz, n_samples, sph, dsph, reinterpret_cast<void*>(stream));
}

template <template <typename> class C, typename T>
inline void cuda_sph_with_hessians(
    cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len
) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    T* sph = reinterpret_cast<T*>(in[1]);
    T* dsph = reinterpret_cast<T*>(in[2]);
    T* ddsph = reinterpret_cast<T*>(in[3]);

    const SphDescriptor& d = *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;

    auto& calculator = _get_or_create_sph_cuda<C, T>(lmax);
    calculator->compute_with_hessians(
        xyz, n_samples, sph, dsph, ddsph, reinterpret_cast<void*>(stream)
    );
}

void cuda_spherical_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph<sphericart::cuda::SphericalHarmonics, float>(stream, in, opaque, opaque_len);
}

void cuda_spherical_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph<sphericart::cuda::SphericalHarmonics, double>(stream, in, opaque, opaque_len);
}

void cuda_dspherical_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_gradients<sphericart::cuda::SphericalHarmonics, float>(
        stream, in, opaque, opaque_len
    );
}

void cuda_dspherical_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_gradients<sphericart::cuda::SphericalHarmonics, double>(
        stream, in, opaque, opaque_len
    );
}

void cuda_ddspherical_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_hessians<sphericart::cuda::SphericalHarmonics, float>(
        stream, in, opaque, opaque_len
    );
}

void cuda_ddspherical_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_hessians<sphericart::cuda::SphericalHarmonics, double>(
        stream, in, opaque, opaque_len
    );
}

void cuda_solid_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph<sphericart::cuda::SolidHarmonics, float>(stream, in, opaque, opaque_len);
}

void cuda_solid_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph<sphericart::cuda::SolidHarmonics, double>(stream, in, opaque, opaque_len);
}

void cuda_dsolid_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_gradients<sphericart::cuda::SolidHarmonics, float>(stream, in, opaque, opaque_len);
}

void cuda_dsolid_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_gradients<sphericart::cuda::SolidHarmonics, double>(stream, in, opaque, opaque_len);
}

void cuda_ddsolid_f32(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_hessians<sphericart::cuda::SolidHarmonics, float>(stream, in, opaque, opaque_len);
}

void cuda_ddsolid_f64(cudaStream_t stream, void** in, const char* opaque, std::size_t opaque_len) {
    cuda_sph_with_hessians<sphericart::cuda::SolidHarmonics, double>(stream, in, opaque, opaque_len);
}

} // namespace cuda
} // namespace sphericart_jax
