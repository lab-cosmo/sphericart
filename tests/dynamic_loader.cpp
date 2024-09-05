#include <iostream>
#include <dlfcn.h>
#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <string>

#define CUDA_DLOPEN_SAFE_CALL(func_call)                                                           \
    do {                                                                                           \
        if (!func_call) {                                                                          \
            std::cerr << "CUDA function " #func_call " not loaded\n";                              \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

class CudaDynamicLoader {
  public:
    CudaDynamicLoader() {
        loadCudaLibrary();
        loadNvrtcLibrary();
    }

    ~CudaDynamicLoader() {
        if (cudaLib)
            dlclose(cudaLib);
        if (nvrtcLib)
            dlclose(nvrtcLib);
    }

    // Wrapper for cuInit
    CUresult cuInit() { return p_cuInit(0); }

    // Wrapper for nvrtcCreateProgram
    nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name) {
        return p_nvrtcCreateProgram(prog, src, name, 0, nullptr, nullptr);
    }

    // Wrapper for nvrtcCompileProgram
    nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, const std::vector<std::string>& options) {
        std::vector<const char*> c_options;
        c_options.reserve(options.size());
        for (const auto& option : options) {
            c_options.push_back(option.c_str());
        }
        return p_nvrtcCompileProgram(prog, c_options.size(), c_options.data());
    }

  private:
    void* cudaLib = nullptr;
    void* nvrtcLib = nullptr;

    // Pointers to dynamically loaded functions
    CUresult (*p_cuInit)(unsigned int) = nullptr;
    nvrtcResult (*p_nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char**, const char**) =
        nullptr;
    nvrtcResult (*p_nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;

    // Function to load the CUDA Driver library (libcuda.so)
    void loadCudaLibrary() {
        cudaLib = dlopen("libcuda.so", RTLD_NOW);
        if (!cudaLib) {
            std::cerr << "Failed to load libcuda.so: "
                      << dlerror() << ". Please run the command: find /usr -name libcuda.so and add the directory to you $LD_LIBRARY_PATH"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        p_cuInit = (CUresult(*)(unsigned int))dlsym(cudaLib, "cuInit");
        CUDA_DLOPEN_SAFE_CALL(p_cuInit);
    }

    // Function to load the NVRTC library (libnvrtc.so)
    void loadNvrtcLibrary() {
        nvrtcLib = dlopen("libnvrtc.so", RTLD_NOW);
        if (!nvrtcLib) {
            std::cerr << "Failed to load libnvrtc.so: " << dlerror() << std::endl;
            exit(EXIT_FAILURE);
        }
        p_nvrtcCreateProgram =
            (nvrtcResult(*)(nvrtcProgram*, const char*, const char*, int, const char**, const char**)
            )dlsym(nvrtcLib, "nvrtcCreateProgram");
        CUDA_DLOPEN_SAFE_CALL(p_nvrtcCreateProgram);

        p_nvrtcCompileProgram = (nvrtcResult(*)(nvrtcProgram, int, const char* const*)
        )dlsym(nvrtcLib, "nvrtcCompileProgram");
        CUDA_DLOPEN_SAFE_CALL(p_nvrtcCompileProgram);
    }
};

int main() {
    // Dynamically load and use CUDA and NVRTC
    CudaDynamicLoader cudaLoader;

    // Call CUDA functions via dynamic loading
    CUresult init_result = cudaLoader.cuInit();
    if (init_result != CUDA_SUCCESS) {
        std::cerr << "cuInit failed with error code: " << init_result << std::endl;
        return EXIT_FAILURE;
    }

    nvrtcProgram prog;
    const std::string kernel_code = "__global__ void kernel() {}";
    nvrtcResult createResult =
        cudaLoader.nvrtcCreateProgram(&prog, kernel_code.c_str(), "kernel.cu");
    if (createResult != NVRTC_SUCCESS) {
        std::cerr << "Failed to create NVRTC program" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> compile_options;
    nvrtcResult compileResult = cudaLoader.nvrtcCompileProgram(prog, compile_options);
    if (compileResult != NVRTC_SUCCESS) {
        std::cerr << "Failed to compile NVRTC program" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "CUDA initialized and program compiled successfully!" << std::endl;

    return 0;
}