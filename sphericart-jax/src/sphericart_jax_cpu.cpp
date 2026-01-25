// Typed JAX FFI handlers for sphericart on CPU.
//
// These functions are registered from Python using ctypes + jax.ffi.pycapsule,
// following the JAX FFI tutorial:
// https://docs.jax.dev/en/latest/ffi.html

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "sphericart.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

template <template <typename> class C, typename T>
using CacheMapCPU = std::map<size_t, std::unique_ptr<C<T>>>;

template <template <typename> class C, typename T>
std::unique_ptr<C<T>>& GetOrCreateCPU(size_t l_max) {
    static CacheMapCPU<C, T> cache;
    static std::mutex cache_mutex;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(l_max);
    if (it == cache.end()) {
        it = cache.insert({l_max, std::make_unique<C<T>>(l_max)}).first;
    }
    return it->second;
}

template <ffi::DataType DT>
ffi::Error GetNSamples(const ffi::Buffer<DT>& xyz, int64_t* n_samples_out) {
    auto dims = xyz.dimensions();
    if (dims.size() == 0) {
        return ffi::Error::InvalidArgument("xyz must be an array");
    }
    if (dims.back() != 3) {
        return ffi::Error::InvalidArgument("xyz last dimension must have size 3");
    }
    const int64_t n_samples = static_cast<int64_t>(xyz.element_count() / 3);
    if (n_samples < 0) {
        return ffi::Error::InvalidArgument("invalid xyz shape");
    }
    *n_samples_out = n_samples;
    return ffi::Error::Success();
}

template <template <typename> class C, typename T, ffi::DataType DT>
ffi::Error CpuSphImpl(int64_t l_max_i64, ffi::Buffer<DT> xyz, ffi::ResultBuffer<DT> sph) {
    if (l_max_i64 < 0) {
        return ffi::Error::InvalidArgument("l_max must be non-negative");
    }
    const size_t l_max = static_cast<size_t>(l_max_i64);

    int64_t n_samples = 0;
    if (auto err = GetNSamples<DT>(xyz, &n_samples); err.failure()) {
        return err;
    }

    const size_t sph_size = (l_max + 1) * (l_max + 1);
    const size_t expected_sph_len = static_cast<size_t>(n_samples) * sph_size;
    if (sph->element_count() != expected_sph_len) {
        return ffi::Error::InvalidArgument("output sph has unexpected size");
    }

    const T* xyz_ptr = reinterpret_cast<const T*>(xyz.typed_data());
    T* sph_ptr = reinterpret_cast<T*>(sph->typed_data());

    const size_t xyz_len = xyz.element_count();
    const size_t sph_len = sph->element_count();

    auto& calculator = GetOrCreateCPU<C, T>(l_max);
    calculator->compute_array(xyz_ptr, xyz_len, sph_ptr, sph_len);
    return ffi::Error::Success();
}

template <template <typename> class C, typename T, ffi::DataType DT>
ffi::Error CpuSphGradImpl(
    int64_t l_max_i64, ffi::Buffer<DT> xyz, ffi::ResultBuffer<DT> sph, ffi::ResultBuffer<DT> dsph
) {
    if (l_max_i64 < 0) {
        return ffi::Error::InvalidArgument("l_max must be non-negative");
    }
    const size_t l_max = static_cast<size_t>(l_max_i64);

    int64_t n_samples = 0;
    if (auto err = GetNSamples<DT>(xyz, &n_samples); err.failure()) {
        return err;
    }

    const size_t sph_size = (l_max + 1) * (l_max + 1);
    const size_t sph_len_expected = static_cast<size_t>(n_samples) * sph_size;
    const size_t dsph_len_expected = sph_len_expected * 3;

    if (sph->element_count() != sph_len_expected) {
        return ffi::Error::InvalidArgument("output sph has unexpected size");
    }
    if (dsph->element_count() != dsph_len_expected) {
        return ffi::Error::InvalidArgument("output dsph has unexpected size");
    }

    const T* xyz_ptr = reinterpret_cast<const T*>(xyz.typed_data());
    T* sph_ptr = reinterpret_cast<T*>(sph->typed_data());
    T* dsph_ptr = reinterpret_cast<T*>(dsph->typed_data());

    const size_t xyz_len = xyz.element_count();
    const size_t sph_len = sph->element_count();
    const size_t dsph_len = dsph->element_count();

    auto& calculator = GetOrCreateCPU<C, T>(l_max);
    calculator->compute_array_with_gradients(xyz_ptr, xyz_len, sph_ptr, sph_len, dsph_ptr, dsph_len);
    return ffi::Error::Success();
}

template <template <typename> class C, typename T, ffi::DataType DT>
ffi::Error CpuSphHessImpl(
    int64_t l_max_i64,
    ffi::Buffer<DT> xyz,
    ffi::ResultBuffer<DT> sph,
    ffi::ResultBuffer<DT> dsph,
    ffi::ResultBuffer<DT> ddsph
) {
    if (l_max_i64 < 0) {
        return ffi::Error::InvalidArgument("l_max must be non-negative");
    }
    const size_t l_max = static_cast<size_t>(l_max_i64);

    int64_t n_samples = 0;
    if (auto err = GetNSamples<DT>(xyz, &n_samples); err.failure()) {
        return err;
    }

    const size_t sph_size = (l_max + 1) * (l_max + 1);
    const size_t sph_len_expected = static_cast<size_t>(n_samples) * sph_size;
    const size_t dsph_len_expected = sph_len_expected * 3;
    const size_t ddsph_len_expected = sph_len_expected * 9;

    if (sph->element_count() != sph_len_expected) {
        return ffi::Error::InvalidArgument("output sph has unexpected size");
    }
    if (dsph->element_count() != dsph_len_expected) {
        return ffi::Error::InvalidArgument("output dsph has unexpected size");
    }
    if (ddsph->element_count() != ddsph_len_expected) {
        return ffi::Error::InvalidArgument("output ddsph has unexpected size");
    }

    const T* xyz_ptr = reinterpret_cast<const T*>(xyz.typed_data());
    T* sph_ptr = reinterpret_cast<T*>(sph->typed_data());
    T* dsph_ptr = reinterpret_cast<T*>(dsph->typed_data());
    T* ddsph_ptr = reinterpret_cast<T*>(ddsph->typed_data());

    const size_t xyz_len = xyz.element_count();
    const size_t sph_len = sph->element_count();
    const size_t dsph_len = dsph->element_count();
    const size_t ddsph_len = ddsph->element_count();

    auto& calculator = GetOrCreateCPU<C, T>(l_max);
    calculator->compute_array_with_hessians(
        xyz_ptr, xyz_len, sph_ptr, sph_len, dsph_ptr, dsph_len, ddsph_ptr, ddsph_len
    );
    return ffi::Error::Success();
}

} // namespace

// ===== CPU spherical harmonics =====

ffi::Error CpuSphericalF32Impl(
    int64_t l_max, ffi::Buffer<ffi::F32> xyz, ffi::ResultBuffer<ffi::F32> sph
) {
    return CpuSphImpl<sphericart::SphericalHarmonics, float, ffi::F32>(l_max, xyz, sph);
}

ffi::Error CpuSphericalF64Impl(
    int64_t l_max, ffi::Buffer<ffi::F64> xyz, ffi::ResultBuffer<ffi::F64> sph
) {
    return CpuSphImpl<sphericart::SphericalHarmonics, double, ffi::F64>(l_max, xyz, sph);
}

ffi::Error CpuDSphericalF32Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F32> xyz,
    ffi::ResultBuffer<ffi::F32> sph,
    ffi::ResultBuffer<ffi::F32> dsph
) {
    return CpuSphGradImpl<sphericart::SphericalHarmonics, float, ffi::F32>(l_max, xyz, sph, dsph);
}

ffi::Error CpuDSphericalF64Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F64> xyz,
    ffi::ResultBuffer<ffi::F64> sph,
    ffi::ResultBuffer<ffi::F64> dsph
) {
    return CpuSphGradImpl<sphericart::SphericalHarmonics, double, ffi::F64>(l_max, xyz, sph, dsph);
}

ffi::Error CpuDDSphericalF32Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F32> xyz,
    ffi::ResultBuffer<ffi::F32> sph,
    ffi::ResultBuffer<ffi::F32> dsph,
    ffi::ResultBuffer<ffi::F32> ddsph
) {
    return CpuSphHessImpl<sphericart::SphericalHarmonics, float, ffi::F32>(
        l_max, xyz, sph, dsph, ddsph
    );
}

ffi::Error CpuDDSphericalF64Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F64> xyz,
    ffi::ResultBuffer<ffi::F64> sph,
    ffi::ResultBuffer<ffi::F64> dsph,
    ffi::ResultBuffer<ffi::F64> ddsph
) {
    return CpuSphHessImpl<sphericart::SphericalHarmonics, double, ffi::F64>(
        l_max, xyz, sph, dsph, ddsph
    );
}

// ===== CPU solid harmonics =====

ffi::Error CpuSolidF32Impl(int64_t l_max, ffi::Buffer<ffi::F32> xyz, ffi::ResultBuffer<ffi::F32> sph) {
    return CpuSphImpl<sphericart::SolidHarmonics, float, ffi::F32>(l_max, xyz, sph);
}

ffi::Error CpuSolidF64Impl(int64_t l_max, ffi::Buffer<ffi::F64> xyz, ffi::ResultBuffer<ffi::F64> sph) {
    return CpuSphImpl<sphericart::SolidHarmonics, double, ffi::F64>(l_max, xyz, sph);
}

ffi::Error CpuDSolidF32Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F32> xyz,
    ffi::ResultBuffer<ffi::F32> sph,
    ffi::ResultBuffer<ffi::F32> dsph
) {
    return CpuSphGradImpl<sphericart::SolidHarmonics, float, ffi::F32>(l_max, xyz, sph, dsph);
}

ffi::Error CpuDSolidF64Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F64> xyz,
    ffi::ResultBuffer<ffi::F64> sph,
    ffi::ResultBuffer<ffi::F64> dsph
) {
    return CpuSphGradImpl<sphericart::SolidHarmonics, double, ffi::F64>(l_max, xyz, sph, dsph);
}

ffi::Error CpuDDSolidF32Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F32> xyz,
    ffi::ResultBuffer<ffi::F32> sph,
    ffi::ResultBuffer<ffi::F32> dsph,
    ffi::ResultBuffer<ffi::F32> ddsph
) {
    return CpuSphHessImpl<sphericart::SolidHarmonics, float, ffi::F32>(l_max, xyz, sph, dsph, ddsph);
}

ffi::Error CpuDDSolidF64Impl(
    int64_t l_max,
    ffi::Buffer<ffi::F64> xyz,
    ffi::ResultBuffer<ffi::F64> sph,
    ffi::ResultBuffer<ffi::F64> dsph,
    ffi::ResultBuffer<ffi::F64> ddsph
) {
    return CpuSphHessImpl<sphericart::SolidHarmonics, double, ffi::F64>(l_max, xyz, sph, dsph, ddsph);
}

// ===== Exported handler symbols =====

// Single output
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_spherical_f32,
    CpuSphericalF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_spherical_f64,
    CpuSphericalF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_solid_f32,
    CpuSolidF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_solid_f64,
    CpuSolidF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
);

// Two outputs
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dspherical_f32,
    CpuDSphericalF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dspherical_f64,
    CpuDSphericalF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dsolid_f32,
    CpuDSolidF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dsolid_f64,
    CpuDSolidF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
);

// Three outputs
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_ddspherical_f32,
    CpuDDSphericalF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
        .Ret<ffi::Buffer<ffi::F32>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_ddspherical_f64,
    CpuDDSphericalF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
        .Ret<ffi::Buffer<ffi::F64>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_ddsolid_f32,
    CpuDDSolidF32Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
        .Ret<ffi::Buffer<ffi::F32>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_ddsolid_f64,
    CpuDDSolidF64Impl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
        .Ret<ffi::Buffer<ffi::F64>>() // ddsph
);
