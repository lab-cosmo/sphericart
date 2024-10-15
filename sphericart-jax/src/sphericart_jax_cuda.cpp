// This file defines the Python interface to the XLA custom calls on CUDA
// devices. It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include "sphericart_cuda.hpp"
#include "sphericart/pybind11_kernel_helpers.hpp"

struct SphDescriptor {
    std::int64_t n_samples;
    std::int64_t lmax;
};

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
inline void cuda_sph(void* stream, void** in, const char* opaque, std::size_t opaque_len) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    T* sph = reinterpret_cast<T*>(in[1]);

    const SphDescriptor& d = *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;

    auto& calculator = _get_or_create_sph_cuda<C, T>(lmax);

    calculator->compute(xyz, n_samples, sph, stream);
}

template <template <typename> class C, typename T>
inline void cuda_sph_with_gradients(
    void* stream, void** in, const char* opaque, std::size_t opaque_len
) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    T* sph = reinterpret_cast<T*>(in[1]);
    T* dsph = reinterpret_cast<T*>(in[2]);

    const SphDescriptor& d = *UnpackDescriptor<SphDescriptor>(opaque, opaque_len);
    const std::int64_t n_samples = d.n_samples;
    const std::int64_t lmax = d.lmax;

    auto& calculator = _get_or_create_sph_cuda<C, T>(lmax);
    calculator->compute_with_gradients(xyz, n_samples, sph, dsph, stream);
}

template <template <typename> class C, typename T>
inline void cuda_sph_with_hessians(
    void* stream, void** in, const char* opaque, std::size_t opaque_len
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
    calculator->compute_with_hessians(xyz, n_samples, sph, dsph, ddsph, stream);
}

// Registration of the custom calls with pybind11
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cuda_spherical_f32"] =
        EncapsulateFunction(cuda_sph<sphericart::cuda::SphericalHarmonics, float>);
    dict["cuda_spherical_f64"] =
        EncapsulateFunction(cuda_sph<sphericart::cuda::SphericalHarmonics, double>);
    dict["cuda_dspherical_f32"] =
        EncapsulateFunction(cuda_sph_with_gradients<sphericart::cuda::SphericalHarmonics, float>);
    dict["cuda_dspherical_f64"] =
        EncapsulateFunction(cuda_sph_with_gradients<sphericart::cuda::SphericalHarmonics, double>);
    dict["cuda_ddspherical_f32"] =
        EncapsulateFunction(cuda_sph_with_hessians<sphericart::cuda::SphericalHarmonics, float>);
    dict["cuda_ddspherical_f64"] =
        EncapsulateFunction(cuda_sph_with_hessians<sphericart::cuda::SphericalHarmonics, double>);
    dict["cuda_solid_f32"] = EncapsulateFunction(cuda_sph<sphericart::cuda::SolidHarmonics, float>);
    dict["cuda_solid_f64"] = EncapsulateFunction(cuda_sph<sphericart::cuda::SolidHarmonics, double>);
    dict["cuda_dsolid_f32"] =
        EncapsulateFunction(cuda_sph_with_gradients<sphericart::cuda::SolidHarmonics, float>);
    dict["cuda_dsolid_f64"] =
        EncapsulateFunction(cuda_sph_with_gradients<sphericart::cuda::SolidHarmonics, double>);
    dict["cuda_ddsolid_f32"] =
        EncapsulateFunction(cuda_sph_with_hessians<sphericart::cuda::SolidHarmonics, float>);
    dict["cuda_ddsolid_f64"] =
        EncapsulateFunction(cuda_sph_with_hessians<sphericart::cuda::SolidHarmonics, double>);
    return dict;
}

PYBIND11_MODULE(sphericart_jax_cuda, m) {
    m.def("registrations", &Registrations);
    m.def("build_sph_descriptor", [](std::int64_t n_samples, std::int64_t lmax) {
        return PackDescriptor(SphDescriptor{n_samples, lmax});
    });
}

} // namespace cuda
} // namespace sphericart_jax
