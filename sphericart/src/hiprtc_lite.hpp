#ifndef SPHERICART_HIPRTC_LITE_HPP
#define SPHERICART_HIPRTC_LITE_HPP

#include <string>
#include <unordered_map>
#include <vector>

namespace sphericart {
namespace hip {

/*
 * Minimal hipRTC front-end used by the HIP backend of sphericart.
 *
 * This is a port of the runtime-compilation mechanism of symmetrix-xl's HIP
 * backend (libsymmetrix/source/jit_hiprtc.cpp): libhiprtc is resolved with
 * dlopen at runtime — it is never a link-time dependency — so HIP wheels only
 * require the ROCm runtime, and detailed diagnostics can be reported when the
 * library is missing.
 *
 * Unlike symmetrix-xl (whose generated kernels are extern "C"), sphericart's
 * kernels are templates, so the compile step also lowers kernel name
 * expressions (e.g. "spherical_harmonics_kernel<float>") to the mangled names
 * required by hipModuleGetFunction, mirroring what gpu-lite does with NVRTC
 * on the CUDA side.
 */

struct HiprtcInformation {
    bool available = false;
    std::string library;
    std::string reason;
    int major = 0;
    int minor = 0;
};

HiprtcInformation hiprtc_information();

struct HiprtcCompilation {
    std::vector<char> code;
    // name expression (e.g. "spherical_harmonics_kernel<float>") ->
    // mangled kernel name in the code object
    std::unordered_map<std::string, std::string> lowered_names;
    std::string log;
    int major = 0;
    int minor = 0;
};

HiprtcCompilation compile_hip_with_hiprtc(
    const std::string& source,
    const std::string& source_name,
    const std::vector<std::string>& name_expressions,
    const std::vector<std::string>& options
);

} // namespace hip
} // namespace sphericart

#endif
