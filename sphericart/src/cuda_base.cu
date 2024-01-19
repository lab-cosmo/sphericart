#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#define CUDA_DEVICE_PREFIX __device__

#include "cuda_base.hpp"

#define HARDCODED_LMAX 1

/* MASK used for warp reductions */
#define FULL_MASK 0xffffffff

/*
    Computes the index for buffer values which are shared across GRID_DIM_Y
*/
__device__ int get_index(int i) { return i * blockDim.y + threadIdx.y; }

/*
    Clears the shared memory buffers for the spherical harmonics and gradients
   if required.
*/
template <typename scalar_t>
__device__ inline void
clear_buffers(int nelements, scalar_t *sph, scalar_t *dsph_x, scalar_t *dsph_y,
              scalar_t *dsph_z,

              scalar_t *dsph_dxdx, scalar_t *dsph_dxdy, scalar_t *dsph_dxdz,

              scalar_t *dsph_dydx, scalar_t *dsph_dydy, scalar_t *dsph_dydz,

              scalar_t *dsph_dzdx, scalar_t *dsph_dzdy, scalar_t *dsph_dzdz,
              bool requires_grad, bool requires_hessian) {
    for (int i = threadIdx.x; i < nelements; i += blockDim.x) {
        sph[get_index(i)] = 0.0;

        if (requires_grad) {
            dsph_x[get_index(i)] = 0.0;
            dsph_y[get_index(i)] = 0.0;
            dsph_z[get_index(i)] = 0.0;
        }

        if (requires_hessian) {
            dsph_dxdx[get_index(i)] = 0.0;
            dsph_dxdy[get_index(i)] = 0.0;
            dsph_dxdz[get_index(i)] = 0.0;

            dsph_dydx[get_index(i)] = 0.0;
            dsph_dydy[get_index(i)] = 0.0;
            dsph_dydz[get_index(i)] = 0.0;

            dsph_dzdx[get_index(i)] = 0.0;
            dsph_dzdy[get_index(i)] = 0.0;
            dsph_dzdz[get_index(i)] = 0.0;
        }
    }
    __syncthreads();
}

/*
    Writes out the shared memory buffers to global memory, as well as applying
   normalisation if necessary.
*/
template <typename scalar_t>
__device__ inline void write_buffers(
    size_t edge_idx, size_t nedges, scalar_t x, scalar_t y, scalar_t z,
    scalar_t ir, int n_elements, int offset, scalar_t *buffer_sph,

    scalar_t *buffer_dsph_x, scalar_t *buffer_dsph_y, scalar_t *buffer_dsph_z,

    scalar_t *buffer_dsph_dxdx, scalar_t *buffer_dsph_dxdy,
    scalar_t *buffer_dsph_dxdz,

    scalar_t *buffer_dsph_dydx, scalar_t *buffer_dsph_dydy,
    scalar_t *buffer_dsph_dydz,

    scalar_t *buffer_dsph_dzdx, scalar_t *buffer_dsph_dzdy,
    scalar_t *buffer_dsph_dzdz, scalar_t *sph, scalar_t *dsph, scalar_t *ddsph,
    size_t n_total, bool requires_grad, bool requires_hessian, bool normalize) {
    if (edge_idx < nedges) {
        for (int i = threadIdx.x; i < n_elements; i += blockDim.x) {

            sph[edge_idx * n_total + offset + i] = buffer_sph[get_index(i)];

            if (requires_hessian) {
                auto tmp_dx = buffer_dsph_x[get_index(i)];
                auto tmp_dy = buffer_dsph_y[get_index(i)];
                auto tmp_dz = buffer_dsph_z[get_index(i)];

                auto tmp_dxdx = buffer_dsph_dxdx[get_index(i)];
                auto tmp_dxdy = buffer_dsph_dxdy[get_index(i)];
                auto tmp_dxdz = buffer_dsph_dxdz[get_index(i)];

                auto tmp_dydx = buffer_dsph_dydx[get_index(i)];
                auto tmp_dydy = buffer_dsph_dydy[get_index(i)];
                auto tmp_dydz = buffer_dsph_dydz[get_index(i)];

                auto tmp_dzdx = buffer_dsph_dzdx[get_index(i)];
                auto tmp_dzdy = buffer_dsph_dzdy[get_index(i)];
                auto tmp_dzdz = buffer_dsph_dzdz[get_index(i)];

                if (normalize) {
                    auto tmp = (tmp_dx * x + tmp_dy * y + tmp_dz * z);

                    auto tmpx = x * tmp_dxdx + y * tmp_dydx + z * tmp_dzdx;
                    auto tmpy = x * tmp_dxdy + y * tmp_dydy + z * tmp_dydz;
                    auto tmpz = x * tmp_dxdz + y * tmp_dydz + z * tmp_dzdz;
                    auto tmp2 = x * x * tmp_dxdx + y * y * tmp_dydy +
                                z * z * tmp_dzdz + 2 * x * y * tmp_dxdy +
                                2 * x * z * tmp_dxdz + 2 * y * z * tmp_dydz;

                    tmp_dxdx = (-2 * x * tmpx + tmp_dxdx + 3 * x * x * tmp -
                                tmp - 2 * x * tmp_dx + x * x * tmp2) *
                               (ir * ir);
                    tmp_dydy = (-2 * y * tmpy + tmp_dydy + 3 * y * y * tmp -
                                tmp - 2 * y * tmp_dy + y * y * tmp2) *
                               (ir * ir);
                    tmp_dzdz = (-2 * z * tmpz + tmp_dzdz + 3 * z * z * tmp -
                                tmp - 2 * z * tmp_dz + z * z * tmp2) *
                               (ir * ir);

                    tmp_dxdy = tmp_dydx =
                        (-x * tmpy - y * tmpx + tmp_dxdy + 3 * x * y * tmp -
                         x * tmp_dy - y * tmp_dx + x * y * tmp2) *
                        (ir * ir);
                    tmp_dxdz = tmp_dzdx =
                        (-x * tmpz - z * tmpx + tmp_dxdz + 3 * x * z * tmp -
                         x * tmp_dz - z * tmp_dx + x * z * tmp2) *
                        (ir * ir);
                    tmp_dzdy = tmp_dydz =
                        (-z * tmpy - y * tmpz + tmp_dzdy + 3 * y * z * tmp -
                         z * tmp_dy - y * tmp_dz + y * z * tmp2) *
                        (ir * ir);
                }

                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 0 * n_total +
                      offset + i] = tmp_dxdx;
                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 1 * n_total +
                      offset + i] = tmp_dxdy;
                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 2 * n_total +
                      offset + i] = tmp_dxdz;

                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 0 * n_total +
                      offset + i] = tmp_dydx;
                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 1 * n_total +
                      offset + i] = tmp_dydy;
                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 2 * n_total +
                      offset + i] = tmp_dydz;

                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 0 * n_total +
                      offset + i] = tmp_dzdx;
                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 1 * n_total +
                      offset + i] = tmp_dzdy;
                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 2 * n_total +
                      offset + i] = tmp_dzdz;
            }

            if (requires_grad) {
                auto tmp_dx = buffer_dsph_x[get_index(i)];
                auto tmp_dy = buffer_dsph_y[get_index(i)];
                auto tmp_dz = buffer_dsph_z[get_index(i)];

                // corrects derivatives for normalization
                if (normalize) {
                    auto tmp = (tmp_dx * x + tmp_dy * y + tmp_dz * z);

                    tmp_dx = (tmp_dx - x * tmp) * ir;
                    tmp_dy = (tmp_dy - y * tmp) * ir;
                    tmp_dz = (tmp_dz - z * tmp) * ir;
                }

                dsph[edge_idx * 3 * n_total + 0 * n_total + offset + i] =
                    tmp_dx;
                dsph[edge_idx * 3 * n_total + 1 * n_total + offset + i] =
                    tmp_dy;
                dsph[edge_idx * 3 * n_total + 2 * n_total + offset + i] =
                    tmp_dz;
            }
        }
    }
}

/*
    CUDA kernel for computing Cartesian spherical harmonics and their
   derivatives.
*/
template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    const scalar_t *__restrict__ xyz, int nedges,
    const scalar_t *__restrict__ prefactors, int nprefactors, int lmax,
    int ntotal, bool requires_grad, bool requires_hessian, bool normalize,
    scalar_t *__restrict__ sph, scalar_t *__restrict__ dsph,
    scalar_t *__restrict__ ddsph) {

    extern __shared__ char buffer[];

    size_t offset = 0;

    scalar_t *buffer_c = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_s = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_twomz = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_prefactors = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += nprefactors * sizeof(scalar_t);

    int nl = max(static_cast<int>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
                 2 * lmax + 1);

    scalar_t *buffer_sph = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * nl * sizeof(scalar_t);

    scalar_t *buffer_dsph_x;
    scalar_t *buffer_dsph_y;
    scalar_t *buffer_dsph_z;

    if (requires_grad) {
        buffer_dsph_x = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_y = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_z = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
    }

    scalar_t *buffer_dsph_dxdx;
    scalar_t *buffer_dsph_dxdy;
    scalar_t *buffer_dsph_dxdz;
    scalar_t *buffer_dsph_dydx;
    scalar_t *buffer_dsph_dydy;
    scalar_t *buffer_dsph_dydz;
    scalar_t *buffer_dsph_dzdx;
    scalar_t *buffer_dsph_dzdy;
    scalar_t *buffer_dsph_dzdz;

    if (requires_hessian) {
        buffer_dsph_dxdx = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dxdy = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dxdz = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);

        buffer_dsph_dydx = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dydy = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dydz = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);

        buffer_dsph_dzdx = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dzdy = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
        buffer_dsph_dzdz = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * nl * sizeof(scalar_t);
    }

    size_t edge_idx = blockIdx.x * blockDim.y + threadIdx.y;

    scalar_t x = 0.0;
    scalar_t y = 0.0;
    scalar_t z = 0.0;

    scalar_t x2 = 0.0;
    scalar_t y2 = 0.0;
    scalar_t z2 = 0.0;

    if (threadIdx.y == 0) {
        for (int i = threadIdx.x; i < nprefactors; i += blockDim.x) {
            buffer_prefactors[i] = prefactors[i];
        }
    }
    __syncthreads();

    if (edge_idx < nedges) {
        x = xyz[edge_idx * 3 + 0];
        y = xyz[edge_idx * 3 + 1];
        z = xyz[edge_idx * 3 + 2];

        x2 = x * x;
        y2 = y * y;
        z2 = z * z;
    }

    scalar_t ir = 0.0;

    if (normalize) {
        if (edge_idx < nedges) {
            auto ir2 = 1.0 / (x2 + y2 + z2);
            ir = sqrt(ir2);
            x *= ir;
            y *= ir;
            z *= ir;
            x2 *= ir2;
            y2 *= ir2;
            z2 *= ir2;
        }
    }

    auto rxy = x2 + y2;
    auto twoz = 2 * z;
    if (threadIdx.x == 0) {
        buffer_c[get_index(0)] = 1.0;
        buffer_s[get_index(0)] = 0.0;
        buffer_twomz[get_index(0)] = twoz;

        for (int m = 1; m < lmax + 1; m++) {
            int m_in_idx = get_index(m - 1);
            int m_out_idx = get_index(m);

            scalar_t c = buffer_c[m_in_idx];
            scalar_t s = buffer_s[m_in_idx];
            scalar_t twomz = buffer_twomz[m_in_idx];

            buffer_c[m_out_idx] = c * x - s * y;
            buffer_s[m_out_idx] = c * y + s * x;
            buffer_twomz[m_out_idx] = twomz + twoz;
        }
    }

    __syncthreads();

    // work through hardcoded parts first...
    int ml = min(static_cast<int>(HARDCODED_LMAX), lmax);

    clear_buffers((ml + 1) * (ml + 1), buffer_sph, buffer_dsph_x, buffer_dsph_y,
                  buffer_dsph_z, buffer_dsph_dxdx, buffer_dsph_dxdy,
                  buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
                  buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy,
                  buffer_dsph_dzdz, requires_grad, requires_hessian);

    if (threadIdx.x == 0) {
        if (lmax >= 1) {
            HARDCODED_SPH_MACRO(1, x, y, z, x2, y2, z2, buffer_sph, get_index);
            if (requires_grad) {
                HARDCODED_SPH_DERIVATIVE_MACRO(
                    1, x, y, z, x2, y2, z2, buffer_sph, buffer_dsph_x,
                    buffer_dsph_y, buffer_dsph_z, get_index);
            }

            if (requires_hessian) {
                HARDCODED_SPH_SECOND_DERIVATIVE_MACRO(
                    1, buffer_sph, buffer_dsph_dxdx, buffer_dsph_dxdy,
                    buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
                    buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy,
                    buffer_dsph_dzdz, get_index);
            }
        } else {
            COMPUTE_SPH_L0(buffer_sph, get_index);
            if (requires_grad) {
                COMPUTE_SPH_DERIVATIVE_L0(buffer_sph, buffer_dsph_x,
                                          buffer_dsph_y, buffer_dsph_z,
                                          get_index);

                if (requires_hessian) {
                    COMPUTE_SPH_SECOND_DERIVATIVE_L0(
                        buffer_sph, buffer_dsph_dxdx, buffer_dsph_dxdy,
                        buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
                        buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy,
                        buffer_dsph_dzdz, get_index);
                }
            }
        }
    }
    __syncthreads();

    // write out the values of the hardcoded derivatives from shared memory into
    // global memory.
    write_buffers(
        edge_idx, nedges, x, y, z, ir, (ml + 1) * (ml + 1), 0, buffer_sph,
        buffer_dsph_x, buffer_dsph_y, buffer_dsph_z, buffer_dsph_dxdx,
        buffer_dsph_dxdy, buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
        buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy, buffer_dsph_dzdz,
        sph, dsph, ddsph, ntotal, requires_grad, requires_hessian, normalize);

    // now lets do the generic terms for l > HARDCODED_LMAX
    int size_q = (lmax + 1) * (lmax + 2) / 2;
    int k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;
    scalar_t *qlmk = buffer_prefactors + size_q + k;
    scalar_t *pk = buffer_prefactors + k;
    int base_index = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);

    for (int l = HARDCODED_LMAX + 1; l < lmax + 1; l += 1) {
        int sph_offset = l * blockDim.y;
        /*
            sph_offset needs to point to Y[l, 0], so the mapping from array
           indices to memory locations may look like: sph 0: 0, sph_offset: 0
           sph 1: 0 1 2, sph_offset: 1 sph 2: 0 1 2 3 4, sph_offset: 2 sph 3: 0
           1 2 3 4 5 6, sph_offset: 3 we also need to make sure we select the
           right atom in the buffer, hence multiplication by blockDim.y.
        */

        // clear out temporary storage buffers
        clear_buffers(2 * l + 1, buffer_sph, buffer_dsph_x, buffer_dsph_y,
                      buffer_dsph_z, buffer_dsph_dxdx, buffer_dsph_dxdy,
                      buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
                      buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy,
                      buffer_dsph_dzdz, requires_grad, requires_hessian);

        // Currently only one warp computes the spherical harmonics.
        if (threadIdx.x == 0) {
            if (requires_grad && requires_hessian) {
                generic_sph_l_channel<scalar_t, true, true, HARDCODED_LMAX,
                                      get_index>(
                    l, x, y, z, rxy, pk, qlmk, buffer_c, buffer_s, buffer_twomz,
                    buffer_sph + sph_offset, buffer_dsph_x + sph_offset,
                    buffer_dsph_y + sph_offset, buffer_dsph_z + sph_offset,
                    buffer_dsph_dxdx + sph_offset,
                    buffer_dsph_dxdy + sph_offset,
                    buffer_dsph_dxdz + sph_offset,
                    buffer_dsph_dydx + sph_offset,
                    buffer_dsph_dydy + sph_offset,
                    buffer_dsph_dydz + sph_offset,
                    buffer_dsph_dzdx + sph_offset,
                    buffer_dsph_dzdy + sph_offset,
                    buffer_dsph_dzdz + sph_offset);
            } else if (requires_grad) {
                generic_sph_l_channel<scalar_t, true, false, HARDCODED_LMAX,
                                      get_index>(
                    l, x, y, z, rxy, pk, qlmk, buffer_c, buffer_s, buffer_twomz,
                    buffer_sph + sph_offset, buffer_dsph_x + sph_offset,
                    buffer_dsph_y + sph_offset, buffer_dsph_z + sph_offset,
                    buffer_dsph_dxdx, buffer_dsph_dxdy, buffer_dsph_dxdz,
                    buffer_dsph_dydx, buffer_dsph_dydy, buffer_dsph_dydz,
                    buffer_dsph_dzdx, buffer_dsph_dzdy,
                    buffer_dsph_dzdz // these are nullpointers
                );
            } else {
                generic_sph_l_channel<scalar_t, false, false, HARDCODED_LMAX,
                                      get_index>(
                    l, x, y, z, rxy, pk, qlmk, buffer_c, buffer_s, buffer_twomz,
                    buffer_sph + sph_offset, buffer_dsph_x, buffer_dsph_y,
                    buffer_dsph_z, buffer_dsph_dxdx, buffer_dsph_dxdy,
                    buffer_dsph_dxdz, buffer_dsph_dydx, buffer_dsph_dydy,
                    buffer_dsph_dydz, buffer_dsph_dzdx, buffer_dsph_dzdy,
                    buffer_dsph_dzdz // these are nullpointers
                );
            }
        }

        // write out temporary storage buffers
        write_buffers(edge_idx, nedges, x, y, z, ir, 2 * l + 1, base_index,
                      buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z,
                      buffer_dsph_dxdx, buffer_dsph_dxdy, buffer_dsph_dxdz,
                      buffer_dsph_dydx, buffer_dsph_dydy, buffer_dsph_dydz,
                      buffer_dsph_dzdx, buffer_dsph_dzdy, buffer_dsph_dzdz, sph,
                      dsph, ddsph, ntotal, requires_grad, requires_hessian,
                      normalize);

        base_index += 2 * l + 1;
        qlmk += l + 1;
        pk += l + 1;
    }
}

/*
    Computes the total amount of shared memory space required by
   spherical_harmonics_kernel.

    For lmax <= HARCODED_LMAX, we need to store all (HARDCODED_LMAX + 1)**2
   scalars in shared memory. For lmax > HARDCODED_LMAX, we only need to store
   each spherical harmonics vector per sample in shared memory.
*/
static size_t total_buffer_size(size_t l_max, size_t GRID_DIM_X,
                                size_t GRID_DIM_Y, size_t dtype_size,
                                bool requires_grad, bool requires_hessian) {
    int nl =
        max(static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
            2 * l_max + 1);

    size_t total_buff_size = 0;

    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size; // buffer_c
    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size; // buffer_s
    total_buff_size += GRID_DIM_Y * (l_max + 1) * dtype_size; // buffer_twomz
    total_buff_size +=
        (l_max + 1) * (l_max + 2) * dtype_size;      // buffer_prefactors
    total_buff_size += GRID_DIM_Y * nl * dtype_size; // buffer_sph_out

    if (requires_grad) {
        total_buff_size +=
            3 * GRID_DIM_Y * nl * dtype_size; // buffer_sph_derivs
    }

    if (requires_hessian) {
        total_buff_size +=
            9 * GRID_DIM_Y * nl * dtype_size; // buffer_sph_hessian
    }

    return total_buff_size;
}

/*
    The default shared memory space on most recent NVIDIA cards is defaulted
   49152 bytes, regarldess if there is more available per SM. This method
   attempts to adjust the shared memory to fit the requested configuration if
   the kernel launch parameters exceeds the default 49152 bytes.
*/

bool sphericart::cuda::adjust_cuda_shared_memory(
    size_t element_size, int64_t l_max, int64_t GRID_DIM_X, int64_t GRID_DIM_Y,
    bool requires_grad, bool requires_hessian) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    auto required_buff_size =
        total_buffer_size(l_max, GRID_DIM_X, GRID_DIM_Y, element_size,
                          requires_grad, requires_hessian);

    bool accepted = required_buff_size <= deviceProp.sharedMemPerBlockOptin;

    if (!accepted) {
        std::cerr << "Warning: requested shared memory buffer ("
                  << required_buff_size;
        std::cerr << ") exceeds max available ("
                  << deviceProp.sharedMemPerBlockOptin;
        std::cerr << ") on device " << deviceProp.name << std::endl;
    } else {
        switch (element_size) {
        case 8:
            cudaFuncSetAttribute(spherical_harmonics_kernel<double>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 required_buff_size);
            break;
        case 4:
            cudaFuncSetAttribute(spherical_harmonics_kernel<float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 required_buff_size);
            break;
        }
    }
    return accepted;
}

/*
    Wrapper to launch the CUDA kernel. Returns a vector containing the spherical
   harmonics and their gradients if required, otherwise returns the spherical
   harmonics and an empty tensor.

    GRID_DIM_X is the number of threads to launch in the x dimension. Used to
   parallelize over the sample dimension. GRID_DIM_Y is the number of threads to
   launch in the y dimension. Used only to improve memory throughput on reads
   and writes.

    Total number of threads used is GRID_DIM_X * GRID_DIM_Y.

    cuda_stream should be of type (void *), therefore if you want to pass in
    a cudaStream_t, first do void * stream_ptr = reinterpret_cast<void *>
   (stream);
*/

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_cuda_base(
    const scalar_t *__restrict__ xyz, const int nedges,
    const scalar_t *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    scalar_t *__restrict__ sph, scalar_t *__restrict__ dsph,
    scalar_t *__restrict__ ddsph, void *cuda_stream) {

    int n_total = (l_max + 1) * (l_max + 1);

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) {
        return (x + bdim - 1) / bdim;
    };

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    dim3 block_dim(find_num_blocks(nedges, GRID_DIM_Y));

    size_t total_buff_size = total_buffer_size(
        l_max, GRID_DIM_X, GRID_DIM_Y, sizeof(scalar_t), gradients, hessian);

    spherical_harmonics_kernel<scalar_t>
        <<<block_dim, grid_dim, total_buff_size, cstream>>>(
            xyz, nedges, prefactors, nprefactors, l_max, n_total, gradients,
            hessian, normalize, sph, dsph, ddsph);

    cudaStreamSynchronize(cstream);
}

template void sphericart::cuda::spherical_harmonics_cuda_base<float>(
    const float *__restrict__ xyz, const int nedges,
    const float *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    float *__restrict__ sph, float *__restrict__ dsph,
    float *__restrict__ ddsph, void *cuda_stream);

template void sphericart::cuda::spherical_harmonics_cuda_base<double>(
    const double *__restrict__ xyz, const int nedges,
    const double *__restrict__ prefactors, const int nprefactors,
    const int64_t l_max, const bool normalize, const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y, const bool gradients, const bool hessian,
    double *__restrict__ sph, double *__restrict__ dsph,
    double *__restrict__ ddsph, void *cuda_stream);

/*
    CUDA kernel to computes the backwards pass for autograd.
*/
template <typename scalar_t>
__global__ void backward_kernel(const scalar_t *__restrict__ dsph,
                                const scalar_t *__restrict__ sph_grad,
                                size_t nedges, size_t n_total,
                                scalar_t *__restrict__ xyz_grad) {

    size_t edge_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int spatial = blockIdx.y;

    scalar_t sum = 0.0;

    if (edge_idx < nedges) {
        // for (int j = threadIdx.x; j < sph_grad.size(1); j += blockDim.x) {
        for (int j = threadIdx.x; j < n_total; j += blockDim.x) {
            // sum += dsph[edge_idx][spatial][j] * sph_grad[edge_idx][j];
            sum += dsph[edge_idx * 3 * n_total + spatial * n_total + j] *
                   sph_grad[edge_idx * n_total + j];
        }
    }

    __syncthreads();

    // reduce across the sub-warp
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (edge_idx < nedges) {
        if (threadIdx.x == 0) {
            // xyz_grad[sample_idx][spatial] = sum;
            xyz_grad[edge_idx * 3 + spatial] = sum;
        }
    }
}

template <typename scalar_t>
void sphericart::cuda::spherical_harmonics_backward_cuda_base(
    const scalar_t *__restrict__ dsph, const scalar_t *__restrict__ sph_grad,
    const int nedges, const int ntotal, scalar_t *__restrict__ xyz_grad,
    void *cuda_stream) {

    dim3 grid_dim(4, 32);

    auto find_num_blocks = [](int x, int bdim) {
        return (x + bdim - 1) / bdim;
    };

    dim3 block_dim(find_num_blocks(nedges, 32), 3);

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    backward_kernel<scalar_t><<<block_dim, grid_dim, 0, cstream>>>(
        dsph, sph_grad, nedges, ntotal, xyz_grad);

    cudaStreamSynchronize(cstream);
}

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<float>(
    const float *__restrict__ dsph, const float *__restrict__ sph_grad,
    const int nedges, const int ntotal, float *__restrict__ xyz_grad,
    void *cuda_stream);

template void sphericart::cuda::spherical_harmonics_backward_cuda_base<double>(
    const double *__restrict__ dsph, const double *__restrict__ sph_grad,
    const int nedges, const int ntotal, double *__restrict__ xyz_grad,
    void *cuda_stream);
