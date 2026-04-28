#include <torch/torch.h>
#include <ATen/DeviceGuard.h>

#include "sphericart/torch.hpp"

#include <cstdint>
#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <map>
#include <memory>
#include <tuple>

using namespace sphericart_torch;

namespace {

class CUDAStream {
  public:
    static CUDAStream& instance() {
        static CUDAStream instance;
        return instance;
    }

    using get_stream_t = void* (*)(uint8_t);
    get_stream_t get_stream = nullptr;

    CUDAStream() {
#ifdef __linux__
        handle = dlopen("libsphericart_torch_cuda_stream.so", RTLD_NOW);
        if (!handle) {
            throw std::runtime_error(
                std::string("Failed to load libsphericart_torch_cuda_stream.so: ") + dlerror()
            );
        }

        auto get_stream = reinterpret_cast<get_stream_t>(dlsym(handle, "get_current_cuda_stream"));
        if (!get_stream) {
            throw std::runtime_error(
                std::string("Failed to load get_current_cuda_stream: ") + dlerror()
            );
        }
        this->get_stream = get_stream;
#elif defined(_WIN32)
        handle = LoadLibraryA("sphericart_torch_cuda_stream.dll");
        if (!handle) {
            throw std::runtime_error("Failed to load sphericart_torch_cuda_stream.dll");
        }

        auto get_stream =
            reinterpret_cast<get_stream_t>(GetProcAddress(handle, "get_current_cuda_stream"));
        if (!get_stream) {
            FreeLibrary(handle);
            throw std::runtime_error("Failed to load get_current_cuda_stream");
        }
        this->get_stream = get_stream;
#else
        throw std::runtime_error("Platform not supported for dynamic loading of CUDA streams");
#endif
    }

    ~CUDAStream() {
#ifdef __linux__
        if (handle) {
            dlclose(handle);
        }
#elif defined(_WIN32)
        if (handle) {
            FreeLibrary(handle);
        }
#endif
    }

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
};

using CacheKey = std::pair<int64_t, int64_t>; // (l_max, device)

template <typename Calculator> using CacheMap = std::map<CacheKey, std::unique_ptr<Calculator>>;

template <typename Calculator> Calculator& get_or_create_calculator(int64_t l_max, int64_t device) {
    thread_local CacheMap<Calculator> cache;
    auto it = cache.find({l_max, device});
    if (it == cache.end()) {
        it = cache.emplace(CacheKey{l_max, device}, std::make_unique<Calculator>(l_max)).first;
    }
    return *(it->second);
}

template <typename Calculator>
std::vector<torch::Tensor> compute_raw(
    torch::Tensor xyz, int64_t l_max, bool do_gradients, bool do_hessians
) {
    xyz = xyz.contiguous();

    if (xyz.device().is_cpu()) {
        auto& calculator = get_or_create_calculator<Calculator>(l_max, -1);
        return calculator.compute_raw_cpu(xyz, do_gradients, do_hessians);
    } else if (xyz.device().is_cuda()) {
        auto& calculator = get_or_create_calculator<Calculator>(l_max, xyz.device().index());
        at::DeviceGuard guard(xyz.device());
        void* stream = CUDAStream::instance().get_stream(xyz.device().index());
        return calculator.compute_raw_cuda(xyz, do_gradients, do_hessians, stream);
    } else {
        throw std::runtime_error("Spherical harmonics are only implemented for CPU and CUDA");
    }
}

template <typename Calculator> torch::Tensor compute(torch::Tensor xyz, int64_t l_max) {
    return compute_raw<Calculator>(xyz, l_max, false, false)[0];
}

template <typename Calculator>
std::tuple<torch::Tensor, torch::Tensor> compute_with_gradients(torch::Tensor xyz, int64_t l_max) {
    auto result = compute_raw<Calculator>(xyz, l_max, true, false);
    return {result[0], result[1]};
}

template <typename Calculator>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_with_hessians(
    torch::Tensor xyz, int64_t l_max
) {
    auto result = compute_raw<Calculator>(xyz, l_max, true, true);
    return {result[0], result[1], result[2]};
}

template <typename Calculator> int64_t omp_num_threads(int64_t l_max) {
    auto& calculator = get_or_create_calculator<Calculator>(l_max, -1);
    return calculator.get_omp_num_threads();
}

template <typename Module> void register_ops(Module& m) {
    m.impl("spherical_harmonics", &compute<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_gradients", &compute_with_gradients<SphericalHarmonics>);
    m.impl("spherical_harmonics_with_hessians", &compute_with_hessians<SphericalHarmonics>);
    m.impl("solid_harmonics", &compute<SolidHarmonics>);
    m.impl("solid_harmonics_with_gradients", &compute_with_gradients<SolidHarmonics>);
    m.impl("solid_harmonics_with_hessians", &compute_with_hessians<SolidHarmonics>);
}

} // namespace

template <template <typename> class C, typename scalar_t>
std::vector<torch::Tensor> _compute_raw_cpu(
    C<scalar_t>& calculator, torch::Tensor xyz, int64_t l_max, bool do_gradients, bool do_hessians
) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cpu()) {
        throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
    }

    if (do_hessians && !do_gradients) {
        throw std::runtime_error("internal error: cannot request hessians without gradients");
    }

    auto n_samples = xyz.sizes()[0];
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph_length = n_samples * (l_max + 1) * (l_max + 1);
    auto dsph_length = n_samples * 3 * (l_max + 1) * (l_max + 1);
    auto ddsph_length = n_samples * 9 * (l_max + 1) * (l_max + 1);
    auto sph = torch::empty({n_samples, (l_max + 1) * (l_max + 1)}, options);

    if (do_hessians) {
        auto dsph = torch::empty({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);
        auto ddsph = torch::empty({n_samples, 3, 3, (l_max + 1) * (l_max + 1)}, options);
        calculator.compute_array_with_hessians(
            xyz.data_ptr<scalar_t>(),
            n_samples * 3,
            sph.data_ptr<scalar_t>(),
            sph_length,
            dsph.data_ptr<scalar_t>(),
            dsph_length,
            ddsph.data_ptr<scalar_t>(),
            ddsph_length
        );
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        auto dsph = torch::empty({n_samples, 3, (l_max + 1) * (l_max + 1)}, options);
        calculator.compute_array_with_gradients(
            xyz.data_ptr<scalar_t>(),
            n_samples * 3,
            sph.data_ptr<scalar_t>(),
            sph_length,
            dsph.data_ptr<scalar_t>(),
            dsph_length
        );
        return {sph, dsph, torch::Tensor()};
    } else {
        calculator.compute_array(
            xyz.data_ptr<scalar_t>(), n_samples * 3, sph.data_ptr<scalar_t>(), sph_length
        );
        return {sph, torch::Tensor(), torch::Tensor()};
    }
}

template <template <typename> class C, typename scalar_t>
std::vector<torch::Tensor> _compute_raw_cuda(
    C<scalar_t>* calculator,
    torch::Tensor xyz,
    int64_t l_max,
    bool do_gradients,
    bool do_hessians,
    void* stream
) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error("internal error: called CUDA version on non-CUDA tensor");
    }

    if (do_hessians && !do_gradients) {
        throw std::runtime_error("internal error: cannot request hessians without gradients");
    }

    auto n_samples = xyz.sizes()[0];
    auto lmtotal = (l_max + 1) * (l_max + 1);
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph = torch::empty({n_samples, lmtotal}, options);

    if (do_hessians) {
        auto dsph = torch::empty({n_samples, 3, lmtotal}, options);
        auto ddsph = torch::empty({n_samples, 3, 3, lmtotal}, options);
        calculator->compute_with_hessians(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            dsph.data_ptr<scalar_t>(),
            ddsph.data_ptr<scalar_t>(),
            stream
        );
        return {sph, dsph, ddsph};
    } else if (do_gradients) {
        auto dsph = torch::empty({n_samples, 3, lmtotal}, options);
        calculator->compute_with_gradients(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            dsph.data_ptr<scalar_t>(),
            stream
        );
        return {sph, dsph, torch::Tensor()};
    } else {
        calculator->compute(
            xyz.data_ptr<scalar_t>(),
            n_samples,
            sph.data_ptr<scalar_t>(),
            reinterpret_cast<void*>(stream)
        );
        return {sph, torch::Tensor(), torch::Tensor()};
    }
}

template <typename Calculator, template <typename> class CPUCalculator>
std::vector<torch::Tensor> compute_raw_cpu_impl(
    Calculator& calculator, torch::Tensor xyz, bool do_gradients, bool do_hessians
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cpu<CPUCalculator, double>(
            calculator.calculator_double_, xyz, calculator.l_max_, do_gradients, do_hessians
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cpu<CPUCalculator, float>(
            calculator.calculator_float_, xyz, calculator.l_max_, do_gradients, do_hessians
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

template <typename Calculator, template <typename> class CUDACalculator>
std::vector<torch::Tensor> compute_raw_cuda_impl(
    Calculator& calculator, torch::Tensor xyz, bool do_gradients, bool do_hessians, void* stream
) {
    if (xyz.dtype() == c10::kDouble) {
        return _compute_raw_cuda<CUDACalculator, double>(
            calculator.calculator_cuda_double_ptr.get(),
            xyz,
            calculator.l_max_,
            do_gradients,
            do_hessians,
            stream
        );
    } else if (xyz.dtype() == c10::kFloat) {
        return _compute_raw_cuda<CUDACalculator, float>(
            calculator.calculator_cuda_float_ptr.get(),
            xyz,
            calculator.l_max_,
            do_gradients,
            do_hessians,
            stream
        );
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

template <template <typename> class CPUCalculator, template <typename> class CUDACalculator>
Harmonics<CPUCalculator, CUDACalculator>::Harmonics(int64_t l_max)
    : l_max_(l_max), calculator_double_(l_max_), calculator_float_(l_max_) {
    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();

    if (torch::cuda::is_available()) {
        this->calculator_cuda_double_ptr = std::make_unique<CUDACalculator<double>>(l_max_);
        this->calculator_cuda_float_ptr = std::make_unique<CUDACalculator<float>>(l_max_);
    }
}

template <template <typename> class CPUCalculator, template <typename> class CUDACalculator>
std::vector<torch::Tensor> Harmonics<CPUCalculator, CUDACalculator>::compute_raw_cpu(
    torch::Tensor xyz, bool do_gradients, bool do_hessians
) {
    return compute_raw_cpu_impl<Harmonics, CPUCalculator>(*this, xyz, do_gradients, do_hessians);
}

template <template <typename> class CPUCalculator, template <typename> class CUDACalculator>
std::vector<torch::Tensor> Harmonics<CPUCalculator, CUDACalculator>::compute_raw_cuda(
    torch::Tensor xyz, bool do_gradients, bool do_hessians, void* stream
) {
    return compute_raw_cuda_impl<Harmonics, CUDACalculator>(
        *this, xyz, do_gradients, do_hessians, stream
    );
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.def("spherical_harmonics(Tensor xyz, int l_max) -> Tensor");
    m.def("spherical_harmonics_with_gradients(Tensor xyz, int l_max) -> (Tensor, Tensor)");
    m.def("spherical_harmonics_with_hessians(Tensor xyz, int l_max) -> (Tensor, Tensor, Tensor)");
    m.def("solid_harmonics(Tensor xyz, int l_max) -> Tensor");
    m.def("solid_harmonics_with_gradients(Tensor xyz, int l_max) -> (Tensor, Tensor)");
    m.def("solid_harmonics_with_hessians(Tensor xyz, int l_max) -> (Tensor, Tensor, Tensor)");
    m.def("spherical_harmonics_omp_num_threads(int l_max) -> int");
    m.def("solid_harmonics_omp_num_threads(int l_max) -> int");
}

TORCH_LIBRARY_IMPL(sphericart_torch, CPU, m) { register_ops(m); }

TORCH_LIBRARY_IMPL(sphericart_torch, CUDA, m) { register_ops(m); }

TORCH_LIBRARY_IMPL(sphericart_torch, CompositeExplicitAutograd, m) {
    m.impl("spherical_harmonics_omp_num_threads", &omp_num_threads<SphericalHarmonics>);
    m.impl("solid_harmonics_omp_num_threads", &omp_num_threads<SolidHarmonics>);
}
