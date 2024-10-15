// This file defines the Python interface to the XLA custom calls on CPU.
// It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include "sphericart.hpp"
#include "sphericart/pybind11_kernel_helpers.hpp"

using namespace sphericart_jax;

namespace {

// CPU section

template <template <typename> class C, typename T>
using CacheMapCPU = std::map<size_t, std::unique_ptr<C<T>>>;

template <template <typename> class C, typename T>
std::unique_ptr<C<T>>& _get_or_create_sph_cpu(size_t l_max) {
    // Static map to cache instances based on parameters
    static CacheMapCPU<C, T> sph_cache;
    static std::mutex cache_mutex;

    // Check if instance exists in cache, if not create and store it
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = sph_cache.find(l_max);
    if (it == sph_cache.end()) {
        it = sph_cache.insert({l_max, std::make_unique<C<T>>(l_max)}).first;
    }
    return it->second;
}

template <template <typename> class C, typename T> void cpu_sph(void* out, const void** in) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    const size_t l_max = *reinterpret_cast<const int*>(in[1]);
    const size_t n_samples = *reinterpret_cast<const int*>(in[2]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    // The output is stored as a single pointer since there is only one output
    T* sph = reinterpret_cast<T*>(out);

    auto& calculator = _get_or_create_sph_cpu<C, T>(l_max);
    calculator->compute_array(xyz, xyz_length, sph, sph_len);
}

template <template <typename> class C, typename T>
void cpu_sph_with_gradients(void* out_tuple, const void** in) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    const size_t l_max = *reinterpret_cast<const int*>(in[1]);
    const size_t n_samples = *reinterpret_cast<const int*>(in[2]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    size_t dsph_len{sph_len * 3};
    // The output is stored as a list of pointers since we have multiple outputs
    void** out = reinterpret_cast<void**>(out_tuple);
    T* sph = reinterpret_cast<T*>(out[0]);
    T* dsph = reinterpret_cast<T*>(out[1]);

    auto& calculator = _get_or_create_sph_cpu<C, T>(l_max);
    calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_len, dsph, dsph_len);
}

template <template <typename> class C, typename T>
void cpu_sph_with_hessians(void* out_tuple, const void** in) {
    // Parse the inputs
    const T* xyz = reinterpret_cast<const T*>(in[0]);
    const size_t l_max = *reinterpret_cast<const int*>(in[1]);
    const size_t n_samples = *reinterpret_cast<const int*>(in[2]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    size_t dsph_len{sph_len * 3};
    size_t ddsph_len{sph_len * 3 * 3};
    // The output is stored as a list of pointers since we have multiple outputs
    void** out = reinterpret_cast<void**>(out_tuple);
    T* sph = reinterpret_cast<T*>(out[0]);
    T* dsph = reinterpret_cast<T*>(out[1]);
    T* ddsph = reinterpret_cast<T*>(out[2]);

    auto& calculator = _get_or_create_sph_cpu<C, T>(l_max);
    calculator->compute_array_with_hessians(
        xyz, xyz_length, sph, sph_len, dsph, dsph_len, ddsph, ddsph_len
    );
}

// Registration of the custom calls with pybind11
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_spherical_f32"] = EncapsulateFunction(cpu_sph<sphericart::SphericalHarmonics, float>);
    dict["cpu_spherical_f64"] = EncapsulateFunction(cpu_sph<sphericart::SphericalHarmonics, double>);
    dict["cpu_dspherical_f32"] =
        EncapsulateFunction(cpu_sph_with_gradients<sphericart::SphericalHarmonics, float>);
    dict["cpu_dspherical_f64"] =
        EncapsulateFunction(cpu_sph_with_gradients<sphericart::SphericalHarmonics, double>);
    dict["cpu_ddspherical_f32"] =
        EncapsulateFunction(cpu_sph_with_hessians<sphericart::SphericalHarmonics, float>);
    dict["cpu_ddspherical_f64"] =
        EncapsulateFunction(cpu_sph_with_hessians<sphericart::SphericalHarmonics, double>);
    dict["cpu_solid_f32"] = EncapsulateFunction(cpu_sph<sphericart::SolidHarmonics, float>);
    dict["cpu_solid_f64"] = EncapsulateFunction(cpu_sph<sphericart::SolidHarmonics, double>);
    dict["cpu_dsolid_f32"] =
        EncapsulateFunction(cpu_sph_with_gradients<sphericart::SolidHarmonics, float>);
    dict["cpu_dsolid_f64"] =
        EncapsulateFunction(cpu_sph_with_gradients<sphericart::SolidHarmonics, double>);
    dict["cpu_ddsolid_f32"] =
        EncapsulateFunction(cpu_sph_with_hessians<sphericart::SolidHarmonics, float>);
    dict["cpu_ddsolid_f64"] =
        EncapsulateFunction(cpu_sph_with_hessians<sphericart::SolidHarmonics, double>);
    return dict;
}

PYBIND11_MODULE(sphericart_jax_cpu, m) { m.def("registrations", &Registrations); }

} // namespace
