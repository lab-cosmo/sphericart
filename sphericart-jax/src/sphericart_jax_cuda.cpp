// This file defines the Python interface to the XLA custom calls on CUDA
// devices. It is exposed as a standard pybind11 module defining "capsule"
// objects containing our methods. For simplicity, we export a separate capsule
// for each supported dtype.
// This file is separated from `sphericart_jax_cuda.cu` because pybind11 does
// not accept cuda files.

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
    dict["cuda_spherical_f32"] = EncapsulateFunction(cuda_spherical_f32);
    dict["cuda_spherical_f64"] = EncapsulateFunction(cuda_spherical_f64);
    dict["cuda_dspherical_f32"] = EncapsulateFunction(cuda_dspherical_f32);
    dict["cuda_dspherical_f64"] = EncapsulateFunction(cuda_dspherical_f64);
    dict["cuda_ddspherical_f32"] = EncapsulateFunction(cuda_ddspherical_f32);
    dict["cuda_ddspherical_f64"] = EncapsulateFunction(cuda_ddspherical_f64);
    dict["cuda_solid_f32"] = EncapsulateFunction(cuda_solid_f32);
    dict["cuda_solid_f64"] = EncapsulateFunction(cuda_solid_f64);
    dict["cuda_dsolid_f32"] = EncapsulateFunction(cuda_dsolid_f32);
    dict["cuda_dsolid_f64"] = EncapsulateFunction(cuda_dsolid_f64);
    dict["cuda_ddsolid_f32"] = EncapsulateFunction(cuda_ddsolid_f32);
    dict["cuda_ddsolid_f64"] = EncapsulateFunction(cuda_ddsolid_f64);
    return dict;
}

PYBIND11_MODULE(sphericart_jax_cuda, m) {
    m.def("registrations", &Registrations);
    m.def("build_sph_descriptor", [](std::int64_t n_samples, std::int64_t lmax) {
        return PackDescriptor(SphDescriptor{n_samples, lmax});
    });
}

} // namespace
