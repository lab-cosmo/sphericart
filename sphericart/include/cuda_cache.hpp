// Cache Manager Header
#ifndef CUDA_CACHE_HPP
#define CUDA_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <iostream>
#include <typeinfo>

#include <dynamic_cuda.hpp>
#include <nvrtc.h>
#include <cuda.h>
#include <cxxabi.h>

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::cerr << "\nerror: " #x " failed with error "                                      \
                      << DynamicCUDA::instance().nvrtcGetErrorString(result) << '\n';              \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                          \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            DynamicCUDA::instance().cuGetErrorName(result, &msg);                                  \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n';                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// TODO demangling below only works for Itanium C++ ABI on Unix-like systems (GNUC or clang)
// Helper function to demangle the type name if necessary
std::string demangleTypeName(const std::string& name) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free
    );
    return (status == 0) ? demangled_name.get() : name;
#else
    throw std::runtime_error("demangling not supported using this toolchain.");
#endif
}

// Base case: No template arguments, return function name without any type information
std::string getKernelName(const std::string& fn_name) { return fn_name; }

// Function to get type name of a single type
template <typename T> std::string typeName() { return demangleTypeName(typeid(T).name()); }

// Variadic template function to build type list
template <typename T, typename... Ts> void buildTemplateTypes(std::string& base) {
    base += typeName<T>(); // Add the first type
    // If there are more types, add a comma and recursively call for the remaining types
    if constexpr (sizeof...(Ts) > 0) {
        base += ", ";
        buildTemplateTypes<Ts...>(base); // Recursively call for the rest of the types
    }
}

// Helper function to start building the types
template <typename T, typename... Ts> std::string buildTemplateTypes() {
    std::string result;
    buildTemplateTypes<T, Ts...>(result); // Use recursive variadic template
    return result;
}

/*
Function to get the kernel name with the list of templated types if any:
*/
template <typename T, typename... Ts> std::string getKernelName(const std::string& fn_name) {
    std::string type_list = buildTemplateTypes<T, Ts...>(); // Build type list
    return fn_name + "<" + type_list + ">"; // Return function name with type list in angle brackets
}

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
        CUDA_SAFE_CALL(DynamicCUDA::instance().cuFuncSetAttribute(function, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        CUDA_SAFE_CALL(DynamicCUDA::instance().cuFuncGetAttribute(&value, attribute, function));
        return value;
    }

    /*
    The default shared memory space on most recent NVIDIA cards is defaulted
   49152 bytes, regarldess if there is more available per SM. This method
   attempts to adjust the shared memory to fit the requested configuration if
   the kernel launch parameters exceeds the default 49152 bytes.
*/
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        /*Check whether we need to adjust shared memory size */
        auto& dynamicCuda = DynamicCUDA::instance();
        if (current_smem_size == 0) {
            CUdevice cuDevice;
            CUresult res = dynamicCuda.cuCtxGetDevice(&cuDevice);

            CUDA_SAFE_CALL(dynamicCuda.cuDeviceGetAttribute(
                &max_smem_size_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice
            ));

            int reserved_smem_per_block = 0;

            CUDA_SAFE_CALL(dynamicCuda.cuDeviceGetAttribute(
                &reserved_smem_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            int curr_max_smem_per_block = 0;

            CUDA_SAFE_CALL(dynamicCuda.cuDeviceGetAttribute(
                &curr_max_smem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            current_smem_size = (curr_max_smem_per_block - reserved_smem_per_block);
        }

        if (query_shared_mem_size > current_smem_size) {

            if (query_shared_mem_size > max_smem_size_optin) {
                throw std::runtime_error(
                    "CachedKernel::launch requested more smem than available on card."
                );
            } else {
                CUDA_SAFE_CALL(dynamicCuda.cuFuncSetAttribute(
                    function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, query_shared_mem_size
                ));
                current_smem_size = query_shared_mem_size;
            }
        }
    }

    void launch(
        dim3 grid,
        dim3 block,
        size_t shared_mem_size,
        void* cuda_stream,
        void** args,
        bool synchronize = false
    ) {

        auto& dynamicCuda = DynamicCUDA::instance();

        CUDA_SAFE_CALL(dynamicCuda.cuCtxSetCurrent(context));

        checkAndAdjustSharedMem(shared_mem_size);

        cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

        CUDA_SAFE_CALL(dynamicCuda.cuLaunchKernel(
            function, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem_size, cstream, args, 0
        ));

        if (synchronize)
            CUDA_SAFE_CALL(dynamicCuda.cuCtxSynchronize());
    }

  private:
    int current_smem_size = 0;
    int max_smem_size_optin = 0;
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
        kernel_cache[kernel_name] = std::make_unique<CachedKernel>(module, fn, ctx);
    }

    bool hasKernel(const std::string& kernel_name) const {
        return kernel_cache.find(kernel_name) != kernel_cache.end();
    }

    CachedKernel* getKernel(const std::string& kernel_name) const {
        auto it = kernel_cache.find(kernel_name);
        if (it != kernel_cache.end()) {
            return it->second.get();
        }
        throw std::runtime_error("Kernel not found in cache.");
    }

  private:
    CudaCacheManager() {}
    CudaCacheManager(const CudaCacheManager&) = delete;
    CudaCacheManager& operator=(const CudaCacheManager&) = delete;

    std::unordered_map<std::string, std::unique_ptr<CachedKernel>> kernel_cache;
};

class KernelFactory {

  public:
    static KernelFactory& instance() {
        static KernelFactory instance;
        return instance;
    }

    CachedKernel* getOrCreateKernel(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_file,
        const std::vector<std::string>& options
    ) {

        if (!cacheManager.hasKernel(kernel_name)) {
            std::string kernel_code = load_cuda_source(source_path);
            // Kernel not found in cache, compile and cache it
            compileAndCacheKernel(kernel_name, kernel_code, source_file, options);
        }

        return cacheManager.getKernel(kernel_name);
    }

  private:
    KernelFactory() : cacheManager(CudaCacheManager::instance()) {}

    // Reference to the singleton instance of CudaCacheManager
    CudaCacheManager& cacheManager;

    void compileAndCacheKernel(
        const std::string& kernel_name,
        const std::string& kernel_code,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        auto& dynamicCuda = DynamicCUDA::instance();
        initCudaDriver();

        CUdevice cuDevice;
        dynamicCuda.cuCtxGetDevice(&cuDevice);

        CUcontext currentContext = nullptr;

        // Get current context
        dynamicCuda.cuCtxGetCurrent(&currentContext);

        if (!currentContext) {
            // workaround for corner case where a primary context exists but is not
            // the current context, seen in multithreaded use-cases
            dynamicCuda.cuDevicePrimaryCtxRetain(&currentContext, cuDevice);
            dynamicCuda.cuCtxSetCurrent(currentContext);
        }

        nvrtcProgram prog;
        NVRTC_SAFE_CALL(dynamicCuda.nvrtcCreateProgram(
            &prog, kernel_code.c_str(), source_name.c_str(), 0, nullptr, nullptr
        ));

        NVRTC_SAFE_CALL(dynamicCuda.nvrtcAddNameExpression(prog, kernel_name.c_str()));

        std::vector<const char*> c_options;
        c_options.reserve(options.size());
        for (const auto& option : options) {
            c_options.push_back(option.c_str());
        }

        int major = 0;
        int minor = 0;
        CUDA_SAFE_CALL(
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
        );
        CUDA_SAFE_CALL(
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
        );
        int arch = major * 10 + minor;
        std::string smbuf = "--gpu-architecture=sm_" + std::to_string(arch);
        std::cout << "Compiling kernels with option: " << smbuf << std::endl;
        c_options.push_back(smbuf.c_str());

        nvrtcResult compileResult =
            dynamicCuda.nvrtcCompileProgram(prog, c_options.size(), c_options.data());
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(dynamicCuda.nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            NVRTC_SAFE_CALL(dynamicCuda.nvrtcGetProgramLog(prog, &log[0]));
            std::cerr << log << std::endl;
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program."
            );
        }

        // Get PTX code
        size_t ptxSize;
        NVRTC_SAFE_CALL(dynamicCuda.nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptxCode(ptxSize);
        NVRTC_SAFE_CALL(dynamicCuda.nvrtcGetPTX(prog, ptxCode.data()));

        CUmodule module;

        CUresult cuResult = dynamicCuda.cuModuleLoadDataEx(&module, ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to load PTX code into CUDA module (error code: "+ std::to_string(cuResult) + ")"
		);
        }

        const char* lowered_name;
        NVRTC_SAFE_CALL(dynamicCuda.nvrtcGetLoweredName(prog, kernel_name.c_str(), &lowered_name));
        CUfunction kernel;
        CUDA_SAFE_CALL(dynamicCuda.cuModuleGetFunction(&kernel, module, lowered_name));

        cacheManager.cacheKernel(kernel_name, module, kernel, currentContext);

        NVRTC_SAFE_CALL(dynamicCuda.nvrtcDestroyProgram(&prog));
    }

    void initCudaDriver() {

        auto& dynamicCuda = DynamicCUDA::instance();

        int deviceCount = 0;
        // Check if CUDA has already been initialized
        CUresult res = dynamicCuda.cuDeviceGetCount(&deviceCount);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = dynamicCuda.cuInit(0);
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
