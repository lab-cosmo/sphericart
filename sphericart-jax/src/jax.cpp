// This file defines the Python interface to the XLA custom calls.
// It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include "sphericart.hpp"
#include "sphericart/jax_cuda.hpp"
#include "sphericart/pybind11_kernel_helpers.h"
#include "sphericart_cuda.hpp"

#include <iostream>

using namespace sphericart_jax;
using namespace std;

namespace {

// CPU section

template <typename T>
using CacheMapCPU =
    std::map<std::tuple<size_t, bool>,
             std::unique_ptr<sphericart::SphericalHarmonics<T>>>;

template <typename T>
std::unique_ptr<sphericart::SphericalHarmonics<T>> &
_get_or_create_sph_cpu(CacheMapCPU<T> &sph_cache, std::mutex &cache_mutex,
                       size_t l_max, bool normalized) {
    // Check if instance exists in cache, if not create and store it
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto key = std::make_tuple(l_max, normalized);
    auto it = sph_cache.find(key);
    if (it == sph_cache.end()) {
        it = sph_cache
                 .insert(
                     {key, std::make_unique<sphericart::SphericalHarmonics<T>>(
                               l_max, normalized)})
                 .first;
    }
    return it->second;
}

template <typename T> void cpu_sph(void *out, const void **in) {
    std::cout << "STILL OK 0" << std::endl;
    // Parse the inputs
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    const size_t l_max = *reinterpret_cast<const int *>(in[1]);
    const bool normalized = *reinterpret_cast<const bool *>(in[2]);
    const size_t n_samples = *reinterpret_cast<const int *>(in[3]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    // The output is stored as a single pointer since there is only one output
    T *sph = reinterpret_cast<T *>(out);

    // Static map to cache instances based on parameters
    static CacheMapCPU<T> sph_cache;
    static std::mutex cache_mutex;

    auto &calculator =
        _get_or_create_sph_cpu(sph_cache, cache_mutex, l_max, normalized);

    std::cout << "STILL OK" << std::endl;
    calculator->compute_array(xyz, xyz_length, sph, sph_len);
    std::cout << "STILL OK 2" << std::endl;
}

template <typename T>
void cpu_sph_with_gradients(void *out_tuple, const void **in) {
    std::cout << "STILL OK 0" << std::endl;
    // Parse the inputs
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    const size_t l_max = *reinterpret_cast<const int *>(in[1]);
    const bool normalized = *reinterpret_cast<const bool *>(in[2]);
    const size_t n_samples = *reinterpret_cast<const int *>(in[3]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    size_t dsph_len{sph_len * 3};
    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    T *sph = reinterpret_cast<T *>(out[0]);
    T *dsph = reinterpret_cast<T *>(out[1]);

    // Static map to cache instances based on parameters
    static CacheMapCPU<T> sph_cache;
    static std::mutex cache_mutex;

    auto &calculator =
        _get_or_create_sph_cpu(sph_cache, cache_mutex, l_max, normalized);
    std::cout << "STILL OK" << std::endl;
    calculator->compute_array_with_gradients(xyz, xyz_length, sph, sph_len,
                                             dsph, dsph_len);
}

template <typename T>
void cpu_sph_with_hessians(void *out_tuple, const void **in) {
    std::cout << "STILL OK 0" << std::endl;
    // Parse the inputs
    const T *xyz = reinterpret_cast<const T *>(in[0]);
    const size_t l_max = *reinterpret_cast<const int *>(in[1]);
    const bool normalized = *reinterpret_cast<const bool *>(in[2]);
    const size_t n_samples = *reinterpret_cast<const int *>(in[3]);
    size_t xyz_length{n_samples * 3};
    size_t sph_len{(l_max + 1) * (l_max + 1) * n_samples};
    size_t dsph_len{sph_len * 3};
    size_t ddsph_len{sph_len * 3 * 3};
    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    T *sph = reinterpret_cast<T *>(out[0]);
    T *dsph = reinterpret_cast<T *>(out[1]);
    T *ddsph = reinterpret_cast<T *>(out[2]);

    // Static map to cache instances based on parameters
    static CacheMapCPU<T> sph_cache;
    static std::mutex cache_mutex;

    auto &calculator =
        _get_or_create_sph_cpu(sph_cache, cache_mutex, l_max, normalized);
    std::cout << "STILL OK" << std::endl;
    calculator->compute_array_with_hessians(xyz, xyz_length, sph, sph_len, dsph,
                                            dsph_len, ddsph, ddsph_len);
}

// Registration of the custom calls with pybind11

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_sph_f32"] = EncapsulateFunction(cpu_sph<float>);
    dict["cpu_sph_f64"] = EncapsulateFunction(cpu_sph<double>);
    dict["cpu_dsph_f32"] = EncapsulateFunction(cpu_sph_with_gradients<float>);
    dict["cpu_dsph_f64"] = EncapsulateFunction(cpu_sph_with_gradients<double>);
    dict["cpu_ddsph_f32"] = EncapsulateFunction(cpu_sph_with_hessians<float>);
    dict["cpu_ddsph_f64"] = EncapsulateFunction(cpu_sph_with_hessians<double>);
    return dict;
}

PYBIND11_MODULE(sphericart_jax, m) {
    m.def("registrations", &Registrations);
}

} // namespace
