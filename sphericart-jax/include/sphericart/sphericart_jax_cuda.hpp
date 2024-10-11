// This file is needed as a workaround for pybind11 not accepting cuda files.
// Note that all the templated functions are split into separate functions so
// that they can be compiled in the `.cu` file.

#ifndef _SPHERICART_JAX_CUDA_HPP_
#define _SPHERICART_JAX_CUDA_HPP_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct SphDescriptor {
    std::int64_t n_samples;
    std::int64_t lmax;
};

namespace sphericart_jax {

namespace cuda {

void cuda_spherical_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_spherical_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_dspherical_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_dspherical_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_ddspherical_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_ddspherical_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_solid_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_solid_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_dsolid_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_dsolid_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_ddsolid_f32(void* stream, void** in, const char* opaque, std::size_t opaque_len);

void cuda_ddsolid_f64(void* stream, void** in, const char* opaque, std::size_t opaque_len);

} // namespace cuda
} // namespace sphericart_jax

#endif
