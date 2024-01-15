
#ifndef _JAX_CUDA_H_
#define _JAX_CUDA_H_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct SphDescriptor {
    std::int64_t n_samples;
    std::int64_t lmax;
    bool normalize;
};

namespace sphericart_jax {

namespace cuda {

void apply_cuda_sph_f32(cudaStream_t stream, void **in, const char *opaque,
                        std::size_t opaque_len);

void apply_cuda_sph_f64(cudaStream_t stream, void **in, const char *opaque,
                        std::size_t opaque_len);

void apply_cuda_sph_with_gradients_f32(cudaStream_t stream, void **in,
                                       const char *opaque,
                                       std::size_t opaque_len);

void apply_cuda_sph_with_gradients_f64(cudaStream_t stream, void **in,
                                       const char *opaque,
                                       std::size_t opaque_len);

void apply_cuda_sph_with_hessians_f32(cudaStream_t stream, void **in,
                                      const char *opaque,
                                      std::size_t opaque_len);

void apply_cuda_sph_with_hessians_f64(cudaStream_t stream, void **in,
                                      const char *opaque,
                                      std::size_t opaque_len);

} // namespace cuda
} // namespace sphericart_jax

#endif