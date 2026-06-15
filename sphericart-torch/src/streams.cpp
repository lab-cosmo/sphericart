#include <cstdint>

#include <torch/torch.h>

#ifdef _WIN32
#define EXPORT_FUNCTION __declspec(dllexport)
#else
#define EXPORT_FUNCTION __attribute__((visibility("default")))
#endif

#ifdef CUDA_AVAILABLE
#include <c10/cuda/CUDAStream.h>

// This function re-expose `c10::cuda::getCurrentCUDAStream` in a way we can
// call through dlopen/dlsym, and is intended to be compiled in a separate
// shared library from the main sphericart_torch one.
//
// The alternative would be to link the main library directly to
// `libtorch_cuda`, but doing so will prevent users from loading
// sphericart_torch when using a CPU-only version of torch.
extern "C" EXPORT_FUNCTION void* sphericart_get_current_cuda_stream(uint8_t device_id) {
    return reinterpret_cast<void*>(c10::cuda::getCurrentCUDAStream(device_id).stream());
}

#else

extern "C" EXPORT_FUNCTION void* sphericart_get_current_cuda_stream(uint8_t device_id) {
    TORCH_WARN_ONCE(
        "Something wrong is happening: trying to get the current CUDA stream, "
        "but this version of sphericart was compiled without CUDA support"
    );
    return nullptr;
}

#endif
