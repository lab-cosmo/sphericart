// Typed JAX FFI handlers for sphericart on CUDA.
//
// This replaces the old pybind11 capsule-based custom call registration with
// typed XLA FFI handler symbols, loaded from Python via ctypes and registered
// using jax.ffi.register_ffi_target(..., platform="CUDA").
//
// See: https://docs.jax.dev/en/latest/ffi.html

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include "dynamic_cuda.hpp"
#include "sphericart_cuda.hpp"

#include <cuda_runtime_api.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

template <template <typename> class C, typename T>
using CacheMapCUDA = std::map<size_t, std::unique_ptr<C<T>>>;

template <template <typename> class C, typename T>
std::unique_ptr<C<T>>& GetOrCreateCUDA(size_t l_max) {
    static CacheMapCUDA<C, T> cache;
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
ffi::Error CudaSphImpl(
    cudaStream_t stream, int64_t l_max_i64, ffi::Buffer<DT> xyz, ffi::ResultBuffer<DT> sph
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
    const size_t expected_sph_len = static_cast<size_t>(n_samples) * sph_size;
    if (sph->element_count() != expected_sph_len) {
        return ffi::Error::InvalidArgument("output sph has unexpected size");
    }

    const T* xyz_ptr = reinterpret_cast<const T*>(xyz.typed_data());
    T* sph_ptr = reinterpret_cast<T*>(sph->typed_data());

    auto& calculator = GetOrCreateCUDA<C, T>(l_max);
    calculator->compute(
        xyz_ptr, static_cast<size_t>(n_samples), sph_ptr, reinterpret_cast<void*>(stream)
    );

    return ffi::Error::Success();
}

template <template <typename> class C, typename T, ffi::DataType DT>
ffi::Error CudaSphGradImpl(
    cudaStream_t stream,
    int64_t l_max_i64,
    ffi::Buffer<DT> xyz,
    ffi::ResultBuffer<DT> sph,
    ffi::ResultBuffer<DT> dsph
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

    auto& calculator = GetOrCreateCUDA<C, T>(l_max);
    calculator->compute_with_gradients(
        xyz_ptr, static_cast<size_t>(n_samples), sph_ptr, dsph_ptr, reinterpret_cast<void*>(stream)
    );

    return ffi::Error::Success();
}

template <template <typename> class C, typename T, ffi::DataType DT>
ffi::Error CudaSphHessImpl(
    cudaStream_t stream,
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

    auto& calculator = GetOrCreateCUDA<C, T>(l_max);
    calculator->compute_with_hessians(
        xyz_ptr,
        static_cast<size_t>(n_samples),
        sph_ptr,
        dsph_ptr,
        ddsph_ptr,
        reinterpret_cast<void*>(stream)
    );

    return ffi::Error::Success();
}

} // namespace

// ===== Exported handler symbols (CUDA) =====

// Single output
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_spherical_f32,
    (CudaSphImpl<sphericart::cuda::SphericalHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_spherical_f64,
    (CudaSphImpl<sphericart::cuda::SphericalHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_solid_f32,
    (CudaSphImpl<sphericart::cuda::SolidHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_solid_f64,
    (CudaSphImpl<sphericart::cuda::SolidHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
);

// Two outputs
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dspherical_f32,
    (CudaSphGradImpl<sphericart::cuda::SphericalHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dspherical_f64,
    (CudaSphGradImpl<sphericart::cuda::SphericalHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dsolid_f32,
    (CudaSphGradImpl<sphericart::cuda::SolidHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dsolid_f64,
    (CudaSphGradImpl<sphericart::cuda::SolidHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
);

// Three outputs
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_ddspherical_f32,
    (CudaSphHessImpl<sphericart::cuda::SphericalHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
        .Ret<ffi::Buffer<ffi::F32>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_ddspherical_f64,
    (CudaSphHessImpl<sphericart::cuda::SphericalHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
        .Ret<ffi::Buffer<ffi::F64>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_ddsolid_f32,
    (CudaSphHessImpl<sphericart::cuda::SolidHarmonics, float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F32>>() // xyz
        .Ret<ffi::Buffer<ffi::F32>>() // sph
        .Ret<ffi::Buffer<ffi::F32>>() // dsph
        .Ret<ffi::Buffer<ffi::F32>>() // ddsph
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_ddsolid_f64,
    (CudaSphHessImpl<sphericart::cuda::SolidHarmonics, double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("l_max")
        .Arg<ffi::Buffer<ffi::F64>>() // xyz
        .Ret<ffi::Buffer<ffi::F64>>() // sph
        .Ret<ffi::Buffer<ffi::F64>>() // dsph
        .Ret<ffi::Buffer<ffi::F64>>() // ddsph
);

// ===== Small C ABI helper for Python (ctypes) =====
extern "C" void sphericart_jax_get_cuda_runtime_version(int* major, int* minor) {
    int ver = 0;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaRuntimeGetVersion(&ver));
    if (major) {
        *major = ver / 1000;
    }
    if (minor) {
        *minor = (ver % 1000) / 10;
    }
}
