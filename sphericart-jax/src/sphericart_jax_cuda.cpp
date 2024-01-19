// This file defines the Python interface to the XLA custom calls on CUDA
// devices. It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#include "sphericart/pybind11_kernel_helpers.hpp"
#include "sphericart/sphericart_jax_cuda.hpp"

using namespace sphericart_jax;
using namespace sphericart_jax::cuda;

namespace {

// Registration of the custom calls with pybind11

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cuda_sph_f32"] = EncapsulateFunction(apply_cuda_sph_f32);
    dict["cuda_sph_f64"] = EncapsulateFunction(apply_cuda_sph_f64);
    dict["cuda_dsph_f32"] =
        EncapsulateFunction(apply_cuda_sph_with_gradients_f32);
    dict["cuda_dsph_f64"] =
        EncapsulateFunction(apply_cuda_sph_with_gradients_f64);
    dict["cuda_ddsph_f32"] =
        EncapsulateFunction(apply_cuda_sph_with_hessians_f32);
    dict["cuda_ddsph_f64"] =
        EncapsulateFunction(apply_cuda_sph_with_hessians_f64);
    return dict;
}

PYBIND11_MODULE(sphericart_jax_cuda, m) {
    m.def("registrations", &Registrations);
    m.def("build_sph_descriptor",
          [](std::int64_t n_samples, std::int64_t lmax, bool normalize) {
              return PackDescriptor(SphDescriptor{n_samples, lmax, normalize});
          });
}

} // namespace
