// Cache Manager Header
#ifndef CUDA_CACHE_HPP
#define CUDA_CACHE_HPP

#include <nvrtc.h>
#include <vector>
#include <unordered_map>
#include <string>

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result)       \
                      << '\n';                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                          \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            cuGetErrorName(result, &msg);                                                          \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n';                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUDA_CALL_CHECK(call)                                                                      \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(cudaStatus) << std::endl;                              \
            cudaDeviceReset();                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// Function to load CUDA source code from a file
std::string load_cuda_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

class CachedKernel {

  public:
    CachedKernel(CUmodule m, CUfunction f, CUcontext c) : module(m), function(f), context(c) {}
    // Default constructor (optional but can be useful)
    CachedKernel() = default;

    // Copy constructor
    CachedKernel(const CachedKernel&) = default;

    // Copy assignment operator
    CachedKernel& operator=(const CachedKernel&) = default;

    inline void setFuncAttribute(CUfunction_attribute attribute, int value) const {
        CUDA_SAFE_CALL(cuFuncSetAttribute(function, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        CUDA_SAFE_CALL(cuFuncGetAttribute(&value, attribute, function));
        return value;
    }

    void launch(
        dim3 grid,
        dim3 block,
        size_t shared_mem_size,
        void* cuda_stream,
        void** args,
        bool synchronize = true
    ) {
        CUDA_SAFE_CALL(cuCtxSetCurrent(context));
        cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

        CUDA_SAFE_CALL(cuLaunchKernel(
            function, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem_size, cstream, args, 0
        ));

        if (synchronize)
            CUDA_SAFE_CALL(cuCtxSynchronize());
    }

  private:
    CUmodule module;
    CUfunction function;
    CUcontext context;
};

class CudaCacheManager {
  public:
    static CudaCacheManager& instance() {
        static CudaCacheManager instance;
        return instance;
    }

    void cacheKernel(const std::string& kernel_name, CUmodule module, CUfunction fn, CUcontext ctx) {
        CachedKernel k(module, fn, ctx);
        kernel_cache[kernel_name] = k;
    }

    bool hasKernel(const std::string& kernel_name) const {
        return kernel_cache.find(kernel_name) != kernel_cache.end();
    }

    CachedKernel getKernel(const std::string& kernel_name) const {
        auto it = kernel_cache.find(kernel_name);
        if (it != kernel_cache.end()) {
            return it->second;
        }
        throw std::runtime_error("Kernel not found in cache.");
    }

  private:
    CudaCacheManager() {}
    CudaCacheManager(const CudaCacheManager&) = delete;
    CudaCacheManager& operator=(const CudaCacheManager&) = delete;

    std::unordered_map<std::string, CachedKernel> kernel_cache;
};

class KernelFactory {

  public:
    static KernelFactory& instance() {
        static KernelFactory instance;
        return instance;
    }

    CachedKernel getOrCreateKernel(
        const std::string& kernel_name, const std::string& source_path, const std::string& source_file
    ) {

        if (!cacheManager.hasKernel(kernel_name)) {
            std::string kernel_code = load_cuda_source(source_path);
            // Kernel not found in cache, compile and cache it
            compileAndCacheKernel(kernel_name, kernel_code, source_file);
        }

        return cacheManager.getKernel(kernel_name);
    }

  private:
    KernelFactory() : cacheManager(CudaCacheManager::instance()) {}

    // Reference to the singleton instance of CudaCacheManager
    CudaCacheManager& cacheManager;

    void compileAndCacheKernel(
        const std::string& kernel_name, const std::string& kernel_code, const std::string& source_name
    ) {

        initCudaDriver();

        CUdevice cuDevice;
        CUresult res = cuCtxGetDevice(&cuDevice);

        CUcontext currentContext;
        // Get current context
        CUresult result = cuCtxGetCurrent(&currentContext);
        // If no context exists, create a new one
        if (res != CUDA_SUCCESS || currentContext == NULL) {
            // Select device (you can modify the device selection logic as needed)

            // Create a new context
            CUresult ctxResult = cuCtxCreate(&currentContext, 0, cuDevice);

            if (ctxResult != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "KernelFactory::compileAndCacheKernel: Failed to create CUDA context."
                );
            }
        }

        nvrtcProgram prog;
        NVRTC_SAFE_CALL(
            nvrtcCreateProgram(&prog, kernel_code.c_str(), source_name.c_str(), 0, nullptr, nullptr)
        );

        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_name.c_str()));

        const char* opts[] = {
            "--include-path=" SPHERICART_INCLUDE_PATH, "--define-macro=CUDA_DEVICE_PREFIX=__device__"
        };

        nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
            std::cerr << log << std::endl;
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program."
            );
        }

        // Get PTX code
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptxCode(ptxSize);
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptxCode.data()));

        CUmodule module;

        CUresult cuResult = cuModuleLoadDataEx(&module, ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to load PTX code into CUDA module"
            );
        }

        const char* lowered_name;
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernel_name.c_str(), &lowered_name));
        CUfunction kernel;
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, lowered_name));

        cacheManager.cacheKernel(kernel_name, module, kernel, currentContext);

        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    }

    void initCudaDriver() {
        int deviceCount = 0;

        // Check if CUDA has already been initialized
        CUresult res = cuDeviceGetCount(&deviceCount);

        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = cuInit(0);
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "KernelFactory::initCudaDriver: Failed to initialize CUDA driver."
                );
                return;
            }
        }
    }

    KernelFactory(const KernelFactory&) = delete;
    KernelFactory& operator=(const KernelFactory&) = delete;
};

#endif