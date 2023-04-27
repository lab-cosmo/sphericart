#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "sphericart/cuda.hpp"

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

#define HARDCODED_LMAX 3
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x " and " #y " must have the same dtype.")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Computes the index for buffer values which are shared across GRID_DIM_Y */
__device__ int get_index(int i) { return i * blockDim.x + threadIdx.x; }

template <typename scalar_t, bool requires_grad>
__device__ void generic_sph_l_channel_device(
    int l,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t rxy,
    scalar_t twoz,
    scalar_t *sph,
    scalar_t *dsph_x,
    scalar_t *dsph_y,
    scalar_t *dsph_z,
    int sph_offset,
    scalar_t *pk,
    scalar_t *qlmk,
    scalar_t *c,
    scalar_t *s
) {
    scalar_t qlm_2, qlm_1, qlm_0;
    scalar_t ql1m_2, ql1m_1, ql1m_0;

    qlm_2 = qlmk[l];

    scalar_t pq = qlm_2 * pk[l];
    scalar_t pdq = 0.0;
    scalar_t pdqx = 0.0;
    scalar_t pdqy = 0.0;

    scalar_t s_l = s[get_index(l)];
    scalar_t s_l_neg1 = s[get_index(l - 1)];
    scalar_t c_l = c[get_index(l)];
    scalar_t c_l_neg1 = c[get_index(l - 1)];

    sph[get_index(sph_offset - l)] = pq * s_l;
    sph[get_index(sph_offset + l)] = pq * c_l;

    if constexpr(requires_grad) {
        pq *= l;
        dsph_x[get_index(sph_offset - l)] = pq * s_l_neg1;
        dsph_y[get_index(sph_offset - l)] = dsph_x[get_index(sph_offset + l)] = pq * c_l_neg1;
        dsph_y[get_index(sph_offset + l)] = -dsph_x[get_index(sph_offset - l)];
        dsph_z[get_index(sph_offset - l)] = 0;
        dsph_z[get_index(sph_offset + l)] = 0;
        ql1m_2 = 0;
    }

    qlm_1 = -z * qlm_2;
    pq = qlm_1 * pk[l - 1];
    sph[get_index(sph_offset - l + 1)] = pq * s_l_neg1;
    sph[get_index(sph_offset + l - 1)] = pq * c_l_neg1;

    if constexpr(requires_grad) {
        pq *= (l - 1);
        dsph_x[get_index(sph_offset - l + 1)] = pq * s[get_index(l - 2)];
        dsph_y[get_index(sph_offset + -l + 1)] = dsph_x[get_index(sph_offset + l - 1)] = pq * c[get_index(l - 2)];
        dsph_y[get_index(sph_offset + l - 1)] = -dsph_x[get_index(sph_offset - l + 1)];

        // uses Q(l-1)(l-1) to initialize the other recursion
        ql1m_1 = qlmk[-1];
        pdq = pk[l - 1] * (l + l - 1) * ql1m_1;
        dsph_z[get_index(sph_offset - l + 1)] = pdq * s_l_neg1;
        dsph_z[get_index(sph_offset + l - 1)] = pdq * c[get_index(l - 1)];
    }

    // and now do the other m's, decrementally
    auto twomz = l * twoz; // compute decrementally to hold 2(m+1)z
    for (auto m = l - 2; m > HARDCODED_LMAX - 1; --m) {
        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];

        auto s_m = s[get_index(m)];
        auto c_m = c[get_index(m)];

        auto s_m_neg1 = s[get_index(m - 1)];
        auto c_m_neg1 = c[get_index(m - 1)];

        sph[get_index(sph_offset - m)] = pq * s_m;
        sph[get_index(sph_offset + m)] = pq * c_m;

        if constexpr(requires_grad) {
            pq *= m;
            ql1m_0 = qlmk[m - l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1;
            ql1m_1 = ql1m_0; // shift

            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dsph_x[get_index(sph_offset - m)] = pdqx * s_m + pq * s_m_neg1;
            dsph_x[get_index(sph_offset + m)] = pdqx * c_m + pq * c_m_neg1;
            pdqy = pdq * y;
            dsph_y[get_index(sph_offset - m)] = pdqy * s_m + pq * c_m_neg1;
            dsph_y[get_index(sph_offset + m)] = pdqy * c_m - pq * s_m_neg1;
            pdq = pk[m] * (l + m) * ql1m_1;
            dsph_z[get_index(sph_offset - m)] = pdq * s_m;
            dsph_z[get_index(sph_offset + m)] = pdq * c_m;
        }
    }

    for (auto m = HARDCODED_LMAX - 1; m > 0; --m) {
        auto s_m = s[get_index(m)];
        auto c_m = c[get_index(m)];

        auto s_m_neg1 = s[get_index(m - 1)];
        auto c_m_neg1 = c[get_index(m - 1)];

        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];
        sph[get_index(sph_offset - m)] = pq * s_m;
        sph[get_index(sph_offset + m)] = pq * c_m;

        if constexpr(requires_grad) {
            pq *= m;
            ql1m_0 = qlmk[m - l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1;
            ql1m_1 = ql1m_0; // shift

            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dsph_x[get_index(sph_offset - m)] = pdqx * s_m + pq * s_m_neg1;
            dsph_x[get_index(sph_offset + m)] = pdqx * c_m + pq * c_m_neg1;
            pdqy = pdq * y;
            dsph_y[get_index(sph_offset - m)] = pdqy * s_m + pq * c_m_neg1;
            dsph_y[get_index(sph_offset + m)] = pdqy * c_m - pq * s_m_neg1;
            pdq = pk[m] * (l + m) * ql1m_1;
            dsph_z[get_index(sph_offset - m)] = pdq * s_m;
            dsph_z[get_index(sph_offset + m)] = pdq * c_m;
        }
    }

    // m=0
    qlm_0 = qlmk[0] * (twoz * qlm_1 + rxy * qlm_2);
    sph[get_index(sph_offset)] = qlm_0 * pk[0];

    if constexpr(requires_grad) {
        ql1m_0 = qlmk[-l] * (twoz * ql1m_1 + rxy * ql1m_2);
        ql1m_2 = ql1m_1;
        ql1m_1 = ql1m_0; // shift
        // derivatives
        dsph_x[get_index(sph_offset)] = pk[0] * x * ql1m_2;
        dsph_y[get_index(sph_offset)] = pk[0] * y * ql1m_2;
        dsph_z[get_index(sph_offset)] = pk[0] * l * ql1m_1;
    }
}

template <typename scalar_t>
__device__ inline void clear_buffers(
    int nelements,
    scalar_t *sph,
    scalar_t *dsph_x,
    scalar_t *dsph_y,
    scalar_t *dsph_z,
    bool requires_grad
) {
    for (int i = 0; i < nelements; i++) {
        sph[get_index(i)] = 0.0;

        if (requires_grad) {
            dsph_x[get_index(i)] = 0.0;
            dsph_y[get_index(i)] = 0.0;
            dsph_z[get_index(i)] = 0.0;
        }
    }
    __syncthreads();
}

template <typename scalar_t>
__device__ inline void write_buffers(
    int atom_idx,
    int natoms,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t ir,
    int n_elements,
    int offset,
    scalar_t *buffer_sph,
    scalar_t *buffer_dsph_x,
    scalar_t *buffer_dsph_y,
    scalar_t *buffer_dsph_z,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph,
    bool requires_grad,
    bool normalize
) {
    if (atom_idx < natoms) {
        for (int i = 0; i < n_elements; i++) {
            sph[atom_idx][offset + i] = buffer_sph[get_index(i)];

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

                dsph[atom_idx][0][offset + i] = tmp_dx;
                dsph[atom_idx][1][offset + i] = tmp_dy;
                dsph[atom_idx][2][offset + i]= tmp_dz;
            }
        }
    }
}

template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> prefactors,
    int lmax,
    bool requires_grad,
    bool normalize,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph
) {
    extern __shared__ char buffer[];

    size_t offset = 0;

    scalar_t *buffer_c = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_s = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_prefactors = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += prefactors.size(0) * sizeof(scalar_t);

    int nl = max(
        static_cast<int>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * lmax + 1
    );

    scalar_t *buffer_sph = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);

    scalar_t *buffer_dsph_x;
    scalar_t *buffer_dsph_y;
    scalar_t *buffer_dsph_z;

    if (requires_grad) {
        buffer_dsph_x = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
        buffer_dsph_y = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
        buffer_dsph_z = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
    }

    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int natoms = xyz.size(0);

    scalar_t x = 0.0;
    scalar_t y = 0.0;
    scalar_t z = 0.0;

    scalar_t x2 = 0.0;
    scalar_t y2 = 0.0;
    scalar_t z2 = 0.0;

    for (int i = threadIdx.x; i < prefactors.size(0); i += blockDim.x) {
        buffer_prefactors[i] = prefactors[i];
    }

    __syncthreads();

    if (atom_idx < natoms) {
        x = xyz[atom_idx][0];
        y = xyz[atom_idx][1];
        z = xyz[atom_idx][2];

        x2 = x * x;
        y2 = y * y;
        z2 = z * z;
    }

    scalar_t ir = 0.0;

    if (normalize) {
        if (atom_idx < natoms) {
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

    auto twoz = 2 * z;
    auto rxy = x2 + y2;

    if (threadIdx.y == 0) {
        buffer_c[get_index(0)] = 1.0;
        buffer_s[get_index(0)] = 0.0;

        for (int m = 1; m < lmax + 1; m++) {
            int m_in_idx = get_index(m - 1);
            int m_out_idx = get_index(m);

            scalar_t c = buffer_c[m_in_idx];
            scalar_t s = buffer_s[m_in_idx];

            buffer_c[m_out_idx] = c * x - s * y;
            buffer_s[m_out_idx] = c * y + s * x;
        }
    }

    __syncthreads();

    // work through hardcoded parts first...
    int ml = min(static_cast<int>(HARDCODED_LMAX), lmax);
    clear_buffers(
        (ml + 1) * (ml + 1),
        buffer_sph,
        buffer_dsph_x,
        buffer_dsph_y,
        buffer_dsph_z,
        requires_grad
    );

    if (lmax>=3) {
        HARDCODED_SPH_MACRO(3, x, y, z, x2, y2, z2, buffer_sph, get_index);
        if (requires_grad) {
            HARDCODED_SPH_DERIVATIVE_MACRO(
                3,
                x, y, z,
                x2, y2, z2,
                buffer_sph,
                buffer_dsph_x,
                buffer_dsph_y,
                buffer_dsph_z,
                get_index
            );
        }
    } else if (lmax>=2) {
        HARDCODED_SPH_MACRO(2, x, y, z, x2, y2, z2, buffer_sph, get_index);
        if (requires_grad) {
            HARDCODED_SPH_DERIVATIVE_MACRO(
                2,
                x, y, z,
                x2, y2, z2,
                buffer_sph,
                buffer_dsph_x,
                buffer_dsph_y,
                buffer_dsph_z,
                get_index
            );
        }
    } else if (lmax>=1) {
        HARDCODED_SPH_MACRO(1, x, y, z, x2, y2, z2, buffer_sph, get_index);
        if (requires_grad) {
            HARDCODED_SPH_DERIVATIVE_MACRO(
                2,
                x, y, z,
                x2, y2, z2,
                buffer_sph,
                buffer_dsph_x,
                buffer_dsph_y,
                buffer_dsph_z,
                get_index
            );
        }
    } else {
        COMPUTE_SPH_L0(buffer_sph, get_index);
        COMPUTE_SPH_DERIVATIVE_L0(buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z, get_index);
    }

    __syncthreads();

    write_buffers(
        atom_idx,
        natoms,
        x,
        y,
        z,
        ir,
        (ml + 1) * (ml + 1),
        0,
        buffer_sph,
        buffer_dsph_x,
        buffer_dsph_y,
        buffer_dsph_z,
        sph,
        dsph,
        requires_grad,
        normalize
    );

    // now lets do the generic terms...
    int size_q = (lmax + 1) * (lmax + 2) / 2;
    int k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;

    scalar_t *qlmk = buffer_prefactors + size_q + k;

    scalar_t *pk = buffer_prefactors + k;

    int base_index = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);

    for (int l = HARDCODED_LMAX + 1; l < lmax + 1; l += 1) {
        int sph_offset = l; // sph needs to point to Y[l, 0]

        // sph 0 : 0
        // sph 1: 0 1 2
        // sph 2: 0 1 2 3 4
        // sph 3: 0 1 2 3 4 5 6

        // clear out temporary storage buffers
        clear_buffers(2 * l + 1, buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z, requires_grad);

        // do some work
        if (requires_grad) {
            generic_sph_l_channel_device<scalar_t, true>(
                l,
                x, y, z,
                rxy, twoz,
                buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z,
                sph_offset,
                pk, qlmk, buffer_c, buffer_s
            );
        } else {
            generic_sph_l_channel_device<scalar_t, false>(
                l,
                x, y, z,
                rxy, twoz,
                buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z,
                sph_offset,
                pk, qlmk, buffer_c, buffer_s
            );
        }

        // write out temporary storage buffers
        write_buffers(
            atom_idx,
            natoms,
            x,
            y,
            z,
            ir,
            2 * l + 1,
            base_index,
            buffer_sph,
            buffer_dsph_x,
            buffer_dsph_y,
            buffer_dsph_z,
            sph,
            dsph,
            requires_grad,
            normalize
        );

        base_index += 2 * l + 1;
        qlmk += l + 1;
        pk += l + 1;
    }
}


static size_t total_buffer_size(size_t l_max, size_t GRID_DIM_X, size_t GRID_DIM_Y, size_t dtype_size, bool requires_grad) {

    int nl = max(
        static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * l_max + 1
     );

    size_t total_buff_size = 0;

    total_buff_size += GRID_DIM_X * (l_max + 1) * dtype_size;      // buffer_c
    total_buff_size += GRID_DIM_X * (l_max + 1) * dtype_size;      // buffer_s
    total_buff_size += (l_max + 1) * (l_max + 2) * dtype_size;     // buffer_prefactors
    total_buff_size += GRID_DIM_Y * GRID_DIM_X * nl * dtype_size;  // buffer_sph_out

    if (requires_grad) {
        total_buff_size += 3 * GRID_DIM_Y * GRID_DIM_X * nl * dtype_size; // buffer_sph_derivs
    }

    return total_buff_size;
}

bool sphericart_torch::adjust_cuda_shared_memory(torch::ScalarType scalar_type, int64_t l_max, int64_t GRID_DIM_X, int64_t GRID_DIM_Y, bool requires_grad) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    size_t dtype = torch::elementSize(scalar_type);
    auto required_buff_size = total_buffer_size(l_max, GRID_DIM_X, GRID_DIM_Y, dtype, requires_grad);

    bool accepted = required_buff_size <= deviceProp.sharedMemPerBlockOptin;

    if (!accepted){
        std::cerr << "Warning: requested shared memory buffer (" << required_buff_size;
        std::cerr << ") exceeds max available (" << deviceProp.sharedMemPerBlockOptin;
        std::cerr << ") on device " << deviceProp.name << std::endl;
    } else {
        switch (scalar_type) {
        case torch::ScalarType::Double:
            cudaFuncSetAttribute(
                spherical_harmonics_kernel<double>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                required_buff_size
            );
            break;
        case torch::ScalarType::Float:
            cudaFuncSetAttribute(
                spherical_harmonics_kernel<float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                required_buff_size
            );
            break;
        // case torch::ScalarType::Half:
        //     cudaFuncSetAttribute(
        //         spherical_harmonics_kernel<at::Half>,
        //         cudaFuncAttributeMaxDynamicSharedMemorySize,
        //         total_buff_size
        //     );
        //     break;
        }
    }
    return accepted;
}

std::vector<torch::Tensor> sphericart_torch::spherical_harmonics_cuda(
    torch::Tensor xyz,
    torch::Tensor prefactors,
    int64_t l_max,
    bool normalize,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool gradients
) {

    CHECK_INPUT(xyz);
    CHECK_INPUT(prefactors);
    CHECK_SAME_DTYPE(xyz, prefactors);

    int n_total = (l_max + 1) * (l_max + 1);

    auto sph = torch::empty(
        {xyz.size(0), n_total},
        torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
    );

    torch::Tensor d_sph;
    if (xyz.requires_grad() || gradients) {
        d_sph = torch::empty(
            {xyz.size(0), 3, n_total},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
        );
    } else {
        // just so accessor doesn't complain
        d_sph = torch::empty(
            {1, 1, 1},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
        );
    }

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 block_dim(find_num_blocks(xyz.size(0), GRID_DIM_X));

    int nl = max(
        static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * l_max + 1
     );

    //int nl = 2 * l_max + 1;

    AT_DISPATCH_FLOATING_TYPES(
        xyz.scalar_type(), "spherical_harmonics_cuda", ([&] {
            size_t total_buff_size = 0;

            total_buff_size += GRID_DIM_X * (l_max + 1) * sizeof(scalar_t);     // buffer_c
            total_buff_size += GRID_DIM_X * (l_max + 1) * sizeof(scalar_t);     // buffer_s
            total_buff_size += (l_max + 1) * (l_max + 2) * sizeof(scalar_t);    // buffer_prefactors
            total_buff_size += GRID_DIM_Y * GRID_DIM_X * nl * sizeof(scalar_t); // buffer_sph_out

            if (xyz.requires_grad() || gradients) {
                total_buff_size += 3 * GRID_DIM_Y * GRID_DIM_X * nl * sizeof(scalar_t); // buffer_sph_derivs
            }

            spherical_harmonics_kernel<<<block_dim, grid_dim, total_buff_size>>>(
                xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                prefactors.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                l_max,
                xyz.requires_grad() || gradients,
                normalize,
                sph.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_sph.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));

    cudaDeviceSynchronize();

    if (xyz.requires_grad() || gradients) {
        return {sph, d_sph};
    } else {
        return {sph, torch::Tensor()};
    }
}

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph_grad,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_grad
) {

        /*
        for (int spatial = 0; spatial < 3; spatial++) {
            auto gradient_slice = dsph.index(
                {torch::indexing::Slice(), spatial, torch::indexing::Slice()}
            );
            xyz_grad.index_put_(
                {torch::indexing::Slice(), spatial},
                torch::sum(sph_grad * gradient_slice, 1)
            );
        }*/

    int atom_idx = blockIdx.x * blockDim.y + threadIdx.y;

    int spatial = blockIdx.y;

    scalar_t sum = 0.0;

    for (int j = threadIdx.x; j < sph_grad.size(1); j +=blockDim.x){
        sum +=  dsph[atom_idx][spatial][j] * sph_grad[atom_idx][j];
    }

    // reduce across the sub-warp
    for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        xyz_grad[atom_idx][spatial]  = sum;
    }

}

torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor dsph,
    torch::Tensor sph_grad
) {

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error("internal error: CUDA version called on non-CUDA tensor");
    }

    auto xyz_grad = torch::Tensor();

    if (xyz.requires_grad()) {
        xyz_grad = torch::empty_like(xyz);

        dim3 grid_dim(4, 32);

        auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

        dim3 block_dim(find_num_blocks(xyz.size(0), 32), 3);

        AT_DISPATCH_FLOATING_TYPES(
        xyz.scalar_type(), "spherical_harmonics_backward_cuda", ([&] {

            backward_kernel<<<block_dim, grid_dim>>>(
                dsph.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                sph_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                xyz_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));

    cudaDeviceSynchronize();

    }

    return xyz_grad;
}

torch::Tensor sphericart_torch::prefactors_cuda(int64_t l_max, at::ScalarType dtype) {
    auto result = torch::empty({(l_max + 1) * (l_max + 2)}, torch::TensorOptions().device("cpu").dtype(dtype));

    if (dtype == c10::kDouble) {
        compute_sph_prefactors(l_max, static_cast<double*>(result.data_ptr()));
    } else if (dtype == c10::kFloat) {
        compute_sph_prefactors(l_max, static_cast<float*>(result.data_ptr()));
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }

    return result.to("cuda");
}
