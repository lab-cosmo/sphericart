// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include <cstdlib>

#include "sphericart.hpp"
#include "sphericart/pybind11_kernel_helpers.h"

using namespace sphericart_jax;

namespace {

template <typename T>
void cpu_sph_with_gradients(void *out_tuple, const void **in) {
  // Parse the inputs
  const size_t l_max = *reinterpret_cast<const size_t *>(in[0]);
  const bool normalized = *reinterpret_cast<const bool *>(in[1]);
  const size_t n_samples = *reinterpret_cast<const size_t *>(in[2]);
  const T *xyz = reinterpret_cast<const T*>(in[3]);
  size_t xyz_length{xyz_length*3};
  size_t sph_len{(l_max+1)*(l_max+1)*n_samples};
  size_t dsph_len{sph_len*3};
  // The output is stored as a list of pointers since we have multiple outputs
  void **out = reinterpret_cast<void **>(out_tuple);
  T *sph = reinterpret_cast<T *>(out[0]);
  T *dsph = reinterpret_cast<T *>(out[1]);

  auto SPH{sphericart::SphericalHarmonics<T>(l_max, normalized)};

  SPH.compute_array_with_gradients(xyz, xyz_length, sph, sph_len, dsph, dsph_len);
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_sph_with_gradients_f32"] = EncapsulateFunction(cpu_sph_with_gradients<float>);
  dict["cpu_sph_with_gradients_f64"] = EncapsulateFunction(cpu_sph_with_gradients<double>);
  return dict;
}

PYBIND11_MODULE(sphericart_jax, m) { m.def("registrations", &Registrations); }

}  // namespace
