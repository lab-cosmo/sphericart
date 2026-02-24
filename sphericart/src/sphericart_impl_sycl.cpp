
#include "sycl_alloc.hpp"
#include "templates_core.hpp"
#include "macros.hpp"
#define HARDCODED_LMAX 1

namespace sphericart {
namespace sycl {
/*
    Computes the index for buffer values which are shared across work-group dimensions
*/
inline int get_index(int i) {
    auto item = syclex::this_work_item::get_nd_item<2>();
    return i * item.get_local_range(1) + item.get_local_id(1);
}

/*
    Clears the shared memory buffers for the spherical harmonics and gradients
   if required.
*/
template <typename scalar_t>
inline void clear_buffers(
    int nelements,
    scalar_t* sph,
    scalar_t* dsph_x,
    scalar_t* dsph_y,
    scalar_t* dsph_z,
    scalar_t* dsph_dxdx,
    scalar_t* dsph_dxdy,
    scalar_t* dsph_dxdz,
    scalar_t* dsph_dydx,
    scalar_t* dsph_dydy,
    scalar_t* dsph_dydz,
    scalar_t* dsph_dzdx,
    scalar_t* dsph_dzdy,
    scalar_t* dsph_dzdz,
    bool requires_grad,
    bool requires_hessian
) {
    auto item = syclex::this_work_item::get_nd_item<2>();
    for (int i = item.get_local_id(0); i < nelements; i += item.get_local_range(0)) {
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
    item.barrier(::sycl::access::fence_space::local_space);
}

/*
    Writes out the shared memory buffers to global memory, as well as applying
   normalisation if necessary.
*/
template <typename scalar_t>
inline void write_buffers(
    int edge_idx,
    int nedges,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t ir,
    int n_elements,
    int offset,
    scalar_t* buffer_sph,
    scalar_t* buffer_dsph_x,
    scalar_t* buffer_dsph_y,
    scalar_t* buffer_dsph_z,
    scalar_t* buffer_dsph_dxdx,
    scalar_t* buffer_dsph_dxdy,
    scalar_t* buffer_dsph_dxdz,
    scalar_t* buffer_dsph_dydx,
    scalar_t* buffer_dsph_dydy,
    scalar_t* buffer_dsph_dydz,
    scalar_t* buffer_dsph_dzdx,
    scalar_t* buffer_dsph_dzdy,
    scalar_t* buffer_dsph_dzdz,
    scalar_t* sph,
    scalar_t* dsph,
    scalar_t* ddsph,
    int n_total,
    bool requires_grad,
    bool requires_hessian,
    bool normalize
) {
    if (edge_idx < nedges) {
        auto item = syclex::this_work_item::get_nd_item<2>();
        for (int i = item.get_local_id(0); i < n_elements; i += item.get_local_range(0)) {
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
                    auto tmpy =
                        x * tmp_dxdy + y * tmp_dydy + z * tmp_dzdy; // * tmp_dydz; TODO correct?
                    auto tmpz = x * tmp_dxdz + y * tmp_dydz + z * tmp_dzdz;
                    auto tmp2 = x * x * tmp_dxdx + y * y * tmp_dydy + z * z * tmp_dzdz +
                                2 * x * y * tmp_dxdy + 2 * x * z * tmp_dxdz + 2 * y * z * tmp_dydz;

                    tmp_dxdx = (-2 * x * tmpx + tmp_dxdx + 3 * x * x * tmp - tmp - 2 * x * tmp_dx +
                                x * x * tmp2) *
                               (ir * ir);
                    tmp_dydy = (-2 * y * tmpy + tmp_dydy + 3 * y * y * tmp - tmp - 2 * y * tmp_dy +
                                y * y * tmp2) *
                               (ir * ir);
                    tmp_dzdz = (-2 * z * tmpz + tmp_dzdz + 3 * z * z * tmp - tmp - 2 * z * tmp_dz +
                                z * z * tmp2) *
                               (ir * ir);

                    tmp_dxdy = tmp_dydx = (-x * tmpy - y * tmpx + tmp_dxdy + 3 * x * y * tmp -
                                           x * tmp_dy - y * tmp_dx + x * y * tmp2) *
                                          (ir * ir);
                    tmp_dxdz = tmp_dzdx = (-x * tmpz - z * tmpx + tmp_dxdz + 3 * x * z * tmp -
                                           x * tmp_dz - z * tmp_dx + x * z * tmp2) *
                                          (ir * ir);
                    tmp_dzdy = tmp_dydz = (-z * tmpy - y * tmpz + tmp_dzdy + 3 * y * z * tmp -
                                           z * tmp_dy - y * tmp_dz + y * z * tmp2) *
                                          (ir * ir);
                }

                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 0 * n_total + offset + i] =
                    tmp_dxdx;
                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 1 * n_total + offset + i] =
                    tmp_dxdy;
                ddsph[edge_idx * 9 * n_total + 0 * 3 * n_total + 2 * n_total + offset + i] =
                    tmp_dxdz;

                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 0 * n_total + offset + i] =
                    tmp_dydx;
                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 1 * n_total + offset + i] =
                    tmp_dydy;
                ddsph[edge_idx * 9 * n_total + 1 * 3 * n_total + 2 * n_total + offset + i] =
                    tmp_dydz;

                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 0 * n_total + offset + i] =
                    tmp_dzdx;
                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 1 * n_total + offset + i] =
                    tmp_dzdy;
                ddsph[edge_idx * 9 * n_total + 2 * 3 * n_total + 2 * n_total + offset + i] =
                    tmp_dzdz;
            }

            if (requires_grad) {
                auto tmp_dx = buffer_dsph_x[get_index(i)];
                auto tmp_dy = buffer_dsph_y[get_index(i)];
                auto tmp_dz = buffer_dsph_z[get_index(i)];

                if (normalize) {
                    auto tmp = (tmp_dx * x + tmp_dy * y + tmp_dz * z);
                    tmp_dx = (tmp_dx - x * tmp) * ir;
                    tmp_dy = (tmp_dy - y * tmp) * ir;
                    tmp_dz = (tmp_dz - z * tmp) * ir;
                }

                dsph[edge_idx * 3 * n_total + 0 * n_total + offset + i] = tmp_dx;
                dsph[edge_idx * 3 * n_total + 1 * n_total + offset + i] = tmp_dy;
                dsph[edge_idx * 3 * n_total + 2 * n_total + offset + i] = tmp_dz;
            }
        }
    }
}

/*
    SYCL kernel for computing Cartesian spherical harmonics and their derivatives.
*/
template <typename scalar_t>
void spherical_harmonics_kernel(
    const scalar_t* xyz_acc,
    int nedges,
    const scalar_t* prefactors,
    int l_max,
    bool requires_grad,
    bool requires_hessian,
    bool normalize,
    scalar_t* sph_acc,
    scalar_t* dsph_acc,
    scalar_t* ddsph_acc
) {
    int ntotal = (l_max + 1) * (l_max + 1);
    int nprefactors = (int)(l_max + 1) * (l_max + 2);

    // Create SYCL queue

    ::sycl::queue& q = *sycl_get_queue();

    // std::cout << "Running on device: " <<
    // sycl_get_queue()->get_device().get_info<::sycl::info::device::name>() << std::endl;

    int GRID_DIM_X = 8, GRID_DIM_Y = 1;
    ::sycl::range<2> local_range(GRID_DIM_X, GRID_DIM_Y);
    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    int groups_y = find_num_blocks(nedges, static_cast<int>(local_range[1]));
    ::sycl::range<2> global_range(local_range[0], groups_y * local_range[1]);

    DEVICE_INIT(scalar_t, prefactors_acc, prefactors, nprefactors);

    int nl = ::sycl::max((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1), 2 * l_max + 1);
    const int local_y = local_range[1];
    const int size_c = local_y * (l_max + 1);
    const int size_sph = local_y * nl;

    // Calculate sizes for local memory allocation
    // We need to use fixed sizes for local memory since they must be known at compile time
    // or passed as kernel parameters
    const int size_grad = requires_grad ? size_sph : 1;
    const int size_hess = requires_hessian ? size_sph : 1;

    q.submit([&](::sycl::handler& cgh) {
        // Allocate work-group local memory for all buffers
        // These are private to each work-group, eliminating the race condition
        ::sycl::local_accessor<scalar_t, 1> buffer_c(::sycl::range<1>(size_c), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_s(::sycl::range<1>(size_c), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_twomz(::sycl::range<1>(size_c), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_prefactors(::sycl::range<1>(nprefactors), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_sph(::sycl::range<1>(size_sph), cgh);

        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_x(::sycl::range<1>(size_grad), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_y(::sycl::range<1>(size_grad), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_z(::sycl::range<1>(size_grad), cgh);

        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dxdx(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dxdy(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dxdz(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dydx(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dydy(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dydz(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dzdx(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dzdy(::sycl::range<1>(size_hess), cgh);
        ::sycl::local_accessor<scalar_t, 1> buffer_dsph_dzdz(::sycl::range<1>(size_hess), cgh);

        cgh.parallel_for(::sycl::nd_range<2>(global_range, local_range), [=](::sycl::nd_item<2> item) {
            // Get raw pointers from local accessors for use in existing functions
            scalar_t* buf_c = buffer_c.get_pointer();
            scalar_t* buf_s = buffer_s.get_pointer();
            scalar_t* buf_twomz = buffer_twomz.get_pointer();
            scalar_t* buf_prefactors = buffer_prefactors.get_pointer();
            scalar_t* buf_sph = buffer_sph.get_pointer();
            scalar_t* buf_dsph_x = buffer_dsph_x.get_pointer();
            scalar_t* buf_dsph_y = buffer_dsph_y.get_pointer();
            scalar_t* buf_dsph_z = buffer_dsph_z.get_pointer();
            scalar_t* buf_dsph_dxdx = buffer_dsph_dxdx.get_pointer();
            scalar_t* buf_dsph_dxdy = buffer_dsph_dxdy.get_pointer();
            scalar_t* buf_dsph_dxdz = buffer_dsph_dxdz.get_pointer();
            scalar_t* buf_dsph_dydx = buffer_dsph_dydx.get_pointer();
            scalar_t* buf_dsph_dydy = buffer_dsph_dydy.get_pointer();
            scalar_t* buf_dsph_dydz = buffer_dsph_dydz.get_pointer();
            scalar_t* buf_dsph_dzdx = buffer_dsph_dzdx.get_pointer();
            scalar_t* buf_dsph_dzdy = buffer_dsph_dzdy.get_pointer();
            scalar_t* buf_dsph_dzdz = buffer_dsph_dzdz.get_pointer();

            int edge_idx = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
            if (item.get_local_id(1) == 0) {
                for (int i = item.get_local_id(0); i < nprefactors; i += item.get_local_range(0)) {
                    buf_prefactors[i] = prefactors_acc[i];
                }
            }
            item.barrier(::sycl::access::fence_space::local_space);

            scalar_t x = 0.0;
            scalar_t y = 0.0;
            scalar_t z = 0.0;
            scalar_t x2 = 0.0;
            scalar_t y2 = 0.0;
            scalar_t z2 = 0.0;

            if (edge_idx < nedges) {
                x = xyz_acc[edge_idx * 3 + 0];
                y = xyz_acc[edge_idx * 3 + 1];
                z = xyz_acc[edge_idx * 3 + 2];
                x2 = x * x;
                y2 = y * y;
                z2 = z * z;
            }

            scalar_t ir = 0.0;

            if (normalize) {
                if (edge_idx < nedges) {
                    auto ir2 = 1.0 / (x2 + y2 + z2);
                    ir = ::sycl::sqrt(ir2);
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

            if (item.get_local_id(0) == 0) {
                buf_c[get_index(0)] = 1.0;
                buf_s[get_index(0)] = 0.0;
                buf_twomz[get_index(0)] = twoz;

                for (int m = 1; m < l_max + 1; m++) {
                    int m_in_idx = get_index(m - 1);
                    int m_out_idx = get_index(m);

                    scalar_t c = buf_c[m_in_idx];
                    scalar_t s = buf_s[m_in_idx];
                    scalar_t twomz_val = buf_twomz[m_in_idx];

                    buf_c[m_out_idx] = c * x - s * y;
                    buf_s[m_out_idx] = c * y + s * x;
                    buf_twomz[m_out_idx] = twomz_val + twoz;
                }
            }
            item.barrier(::sycl::access::fence_space::local_space);

            // work through hardcoded parts first...
            int ml = ::sycl::min(static_cast<int>(HARDCODED_LMAX), l_max);

            clear_buffers(
                (ml + 1) * (ml + 1),
                buf_sph,
                buf_dsph_x,
                buf_dsph_y,
                buf_dsph_z,
                buf_dsph_dxdx,
                buf_dsph_dxdy,
                buf_dsph_dxdz,
                buf_dsph_dydx,
                buf_dsph_dydy,
                buf_dsph_dydz,
                buf_dsph_dzdx,
                buf_dsph_dzdy,
                buf_dsph_dzdz,
                requires_grad,
                requires_hessian
            );
            if (item.get_local_id(0) == 0) {
                if (l_max >= 1) {
                    HARDCODED_SPH_MACRO(1, x, y, z, x2, y2, z2, buf_sph, get_index);
                    if (requires_grad) {
                        HARDCODED_SPH_DERIVATIVE_MACRO(
                            1, x, y, z, x2, y2, z2, buf_sph, buf_dsph_x, buf_dsph_y, buf_dsph_z, get_index
                        );
                    }

                    if (requires_hessian) {
                        HARDCODED_SPH_SECOND_DERIVATIVE_MACRO(
                            1,
                            buf_sph,
                            buf_dsph_dxdx,
                            buf_dsph_dxdy,
                            buf_dsph_dxdz,
                            buf_dsph_dydx,
                            buf_dsph_dydy,
                            buf_dsph_dydz,
                            buf_dsph_dzdx,
                            buf_dsph_dzdy,
                            buf_dsph_dzdz,
                            get_index
                        );
                    }
                } else {
                    COMPUTE_SPH_L0(buf_sph, get_index);
                    if (requires_grad) {
                        COMPUTE_SPH_DERIVATIVE_L0(
                            buf_sph, buf_dsph_x, buf_dsph_y, buf_dsph_z, get_index
                        );

                        if (requires_hessian) {
                            COMPUTE_SPH_SECOND_DERIVATIVE_L0(
                                buf_sph,
                                buf_dsph_dxdx,
                                buf_dsph_dxdy,
                                buf_dsph_dxdz,
                                buf_dsph_dydx,
                                buf_dsph_dydy,
                                buf_dsph_dydz,
                                buf_dsph_dzdx,
                                buf_dsph_dzdy,
                                buf_dsph_dzdz,
                                get_index
                            );
                        }
                    }
                }
            }
            item.barrier(::sycl::access::fence_space::local_space);

            // write out the values of the hardcoded derivatives from local memory into
            // global memory.
            write_buffers(
                edge_idx,
                nedges,
                x,
                y,
                z,
                ir,
                (ml + 1) * (ml + 1),
                0,
                buf_sph,
                buf_dsph_x,
                buf_dsph_y,
                buf_dsph_z,
                buf_dsph_dxdx,
                buf_dsph_dxdy,
                buf_dsph_dxdz,
                buf_dsph_dydx,
                buf_dsph_dydy,
                buf_dsph_dydz,
                buf_dsph_dzdx,
                buf_dsph_dzdy,
                buf_dsph_dzdz,
                sph_acc,
                dsph_acc,
                ddsph_acc,
                ntotal,
                requires_grad,
                requires_hessian,
                normalize
            );

            // Generic spherical harmonics for l > HARDCODED_LMAX
            int size_q = (l_max + 1) * (l_max + 2) / 2;
            int k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;
            scalar_t* qlmk = buf_prefactors + size_q + k;
            scalar_t* pk = buf_prefactors + k;
            int base_index = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);
            for (int l = HARDCODED_LMAX + 1; l < l_max + 1; l += 1) {
                int sph_offset = l * local_y;

                // clear out temporary storage buffers
                clear_buffers(
                    2 * l + 1,
                    buf_sph,
                    buf_dsph_x,
                    buf_dsph_y,
                    buf_dsph_z,
                    buf_dsph_dxdx,
                    buf_dsph_dxdy,
                    buf_dsph_dxdz,
                    buf_dsph_dydx,
                    buf_dsph_dydy,
                    buf_dsph_dydz,
                    buf_dsph_dzdx,
                    buf_dsph_dzdy,
                    buf_dsph_dzdz,
                    requires_grad,
                    requires_hessian
                );
                //
                // Currently only one work-item computes the spherical harmonics.
                if (item.get_local_id(0) == 0) {
                    if (requires_grad && requires_hessian) {
                        generic_sph_l_channel<scalar_t, true, true, HARDCODED_LMAX, get_index>(
                            l,
                            x,
                            y,
                            z,
                            rxy,
                            pk,
                            qlmk,
                            buf_c,
                            buf_s,
                            buf_twomz,
                            buf_sph + sph_offset,
                            buf_dsph_x + sph_offset,
                            buf_dsph_y + sph_offset,
                            buf_dsph_z + sph_offset,
                            buf_dsph_dxdx + sph_offset,
                            buf_dsph_dxdy + sph_offset,
                            buf_dsph_dxdz + sph_offset,
                            buf_dsph_dydx + sph_offset,
                            buf_dsph_dydy + sph_offset,
                            buf_dsph_dydz + sph_offset,
                            buf_dsph_dzdx + sph_offset,
                            buf_dsph_dzdy + sph_offset,
                            buf_dsph_dzdz + sph_offset
                        );
                    } else if (requires_grad) {
                        generic_sph_l_channel<scalar_t, true, false, HARDCODED_LMAX, get_index>(
                            l,
                            x,
                            y,
                            z,
                            rxy,
                            pk,
                            qlmk,
                            buf_c,
                            buf_s,
                            buf_twomz,
                            buf_sph + sph_offset,
                            buf_dsph_x + sph_offset,
                            buf_dsph_y + sph_offset,
                            buf_dsph_z + sph_offset,
                            buf_dsph_dxdx,
                            buf_dsph_dxdy,
                            buf_dsph_dxdz,
                            buf_dsph_dydx,
                            buf_dsph_dydy,
                            buf_dsph_dydz,
                            buf_dsph_dzdx,
                            buf_dsph_dzdy,
                            buf_dsph_dzdz // these are not used
                        );
                    } else {
                        generic_sph_l_channel<scalar_t, false, false, HARDCODED_LMAX, get_index>(
                            l,
                            x,
                            y,
                            z,
                            rxy,
                            pk,
                            qlmk,
                            buf_c,
                            buf_s,
                            buf_twomz,
                            buf_sph + sph_offset,
                            buf_dsph_x,
                            buf_dsph_y,
                            buf_dsph_z,
                            buf_dsph_dxdx,
                            buf_dsph_dxdy,
                            buf_dsph_dxdz,
                            buf_dsph_dydx,
                            buf_dsph_dydy,
                            buf_dsph_dydz,
                            buf_dsph_dzdx,
                            buf_dsph_dzdy,
                            buf_dsph_dzdz // these are not used
                        );
                    }
                }
                ////                // write out temporary storage buffers
                write_buffers(
                    edge_idx,
                    nedges,
                    x,
                    y,
                    z,
                    ir,
                    2 * l + 1,
                    base_index,
                    buf_sph,
                    buf_dsph_x,
                    buf_dsph_y,
                    buf_dsph_z,
                    buf_dsph_dxdx,
                    buf_dsph_dxdy,
                    buf_dsph_dxdz,
                    buf_dsph_dydx,
                    buf_dsph_dydy,
                    buf_dsph_dydz,
                    buf_dsph_dzdx,
                    buf_dsph_dzdy,
                    buf_dsph_dzdz,
                    sph_acc,
                    dsph_acc,
                    ddsph_acc,
                    ntotal,
                    requires_grad,
                    requires_hessian,
                    normalize
                );

                base_index += 2 * l + 1;
                qlmk += l + 1;
                pk += l + 1;
            }
        }); // end parallel_for
    });     // end submit

    q.wait();

    FREE(prefactors_acc);
    // Note: Local memory buffers are automatically deallocated when the kernel completes
    // No need to FREE them as they were allocated using::sycl::local_accessor
}

template void spherical_harmonics_kernel<float>(
    const float* xyz,
    int nedges,
    const float* prefactors,
    int lmax,
    bool requires_grad,
    bool requires_hessian,
    bool normalize,
    float* sph,
    float* dsph,
    float* ddsph
);

template void spherical_harmonics_kernel<double>(
    const double* xyz,
    int nedges,
    const double* prefactors,
    int lmax,
    bool requires_grad,
    bool requires_hessian,
    bool normalize,
    double* sph,
    double* dsph,
    double* ddsph
);

/*
    SYCL kernel to compute the backwards pass for autograd.

    This is the SYCL port of the CUDA backward_kernel, using nd_range<3> to handle:
    - Dimension 0: Reduction dimension (work-items that accumulate partial sums)
    - Dimension 1: Edge/sample dimension (one edge per work-item in Y)
    - Dimension 2: Spatial dimension (3 groups for x, y, z gradients)
*/
template <typename scalar_t>
void backward_kernel(
    const scalar_t* dsph, const scalar_t* sph_grad, int nedges, int n_total, scalar_t* xyz_grad
) {
    ::sycl::queue& q = *sycl_get_queue();

    // Match CUDA configuration: block_dim(4, 32), grid_dim(nedges/32, 3)
    int GRID_DIM_X = 4;  // Reduction dimension (matches CUDA threadIdx.x)
    int GRID_DIM_Y = 32; // Edge dimension (matches CUDA threadIdx.y)
    int GRID_DIM_Z = 1;  // Spatial dimension unit

    ::sycl::range<3> local_range(GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };
    int groups_y = find_num_blocks(nedges, static_cast<int>(local_range[1]));
    int groups_z = 3; // 3 spatial dimensions (x, y, z)

    ::sycl::range<3> global_range(
        local_range[0], groups_y * local_range[1], groups_z * local_range[2]
    );

    // Verify device supports required sub-group size for reduction
    auto sg_sizes = q.get_device().get_info<::sycl::info::device::sub_group_sizes>();
    bool supported = false;
    for (auto size : sg_sizes) {
        if (size >= static_cast<size_t>(GRID_DIM_X)) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        std::string available_sizes;
        for (size_t i = 0; i < sg_sizes.size(); ++i) {
            if (i > 0) {
                available_sizes += ", ";
            }
            available_sizes += std::to_string(sg_sizes[i]);
        }
        throw std::runtime_error(
            "Device does not support required sub-group size for backward_kernel. "
            "Required: " +
            std::to_string(GRID_DIM_X) + ", Available: [" + available_sizes + "]"
        );
    }

    q.submit([&](::sycl::handler& cgh) {
         cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range), [=](::sycl::nd_item<3> item) {
             // Thread indexing (matches CUDA backward_kernel)
             // edge_idx = blockIdx.x * blockDim.y + threadIdx.y
             int edge_idx = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
             // spatial = blockIdx.y (0, 1, or 2 for x, y, z)
             int spatial = item.get_group(2);

             scalar_t sum = 0.0;

             // Accumulation loop (strided by local_range(0) for reduction)
             // Matches CUDA: for (int j = threadIdx.x; j < n_total; j += blockDim.x)
             if (edge_idx < nedges) {
                 for (int j = item.get_local_id(0); j < n_total; j += item.get_local_range(0)) {
                     // sum += dsph[edge_idx][spatial][j] * sph_grad[edge_idx][j]
                     sum += dsph[edge_idx * 3 * n_total + spatial * n_total + j] *
                            sph_grad[edge_idx * n_total + j];
                 }
             }

             item.barrier(::sycl::access::fence_space::local_space);

             // Reduction across dimension 0 using sub-group shuffle operations
             // This is the SYCL equivalent of CUDA's __shfl_down_sync warp reduction
             auto sg = item.get_sub_group();

             // Perform tree reduction using shuffle operations
             // Matches CUDA: for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
             for (int offset = item.get_local_range(0) / 2; offset > 0; offset /= 2) {
                 sum += ::sycl::shift_group_left(sg, sum, offset);
             }

             // Write result (only thread 0 in the reduction dimension writes)
             // Matches CUDA: if (threadIdx.x == 0) xyz_grad[edge_idx * 3 + spatial] = sum
             if (edge_idx < nedges && item.get_local_id(0) == 0) {
                 xyz_grad[edge_idx * 3 + spatial] = sum;
             }
         });
     }).wait();
}

template void backward_kernel<float>(
    const float* dsph, const float* sph_grad, int nedges, int n_total, float* xyz_grad
);

template void backward_kernel<double>(
    const double* dsph, const double* sph_grad, int nedges, int n_total, double* xyz_grad
);
} // namespace sycl
} // namespace sphericart
