// Cache Manager Header
#ifndef SPHERICART_CUDA_CACHE_HPP
#define SPHERICART_CUDA_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <typeinfo>

#include <nvrtc.h>
#include <cuda.h>
#include <cxxabi.h>

#include "dynamic_cuda.hpp"

// Helper function to demangle the type name if necessary
std::string demangleTypeName(const std::string& name) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* undecorated_name std::unique_ptr<char, void (*)(void*)> demangled_name(
        abi::__cxa_demangle(name.c_str(), 0, 0, &status), std::free
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

/*
Container class for the cached kernels. Provides functionality for launching compiled kernels as
well as automatically resizing dynamic shared memory allocations, when needed. Kernels are compiled
on first launch.
*/
class CachedKernel {

  public:
    CachedKernel(
        std::string kernel_name,
        std::string kernel_code,
        std::string source_name,
        std::vector<std::string> options
    ) {
        this->kernel_name = kernel_name;
        this->kernel_code = kernel_code;
        this->source_name = source_name;
        this->options = options;
    }

    CachedKernel() = default;

    void set(CUmodule m, CUfunction f, CUcontext c) {
        this->module = m;
        this->function = f;
        this->context = c;
    }
    // Copy constructor
    CachedKernel(const CachedKernel&) = default;

    // Copy assignment operator
    CachedKernel& operator=(const CachedKernel&) = default;

    inline void setFuncAttribute(CUfunction_attribute attribute, int value) const {
        CUDADRIVER_SAFE_CALL(CUDADriver::instance().cuFuncSetAttribute(function, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        CUDADRIVER_SAFE_CALL(CUDADriver::instance().cuFuncGetAttribute(&value, attribute, function));
        return value;
    }

    /*
    launches the kernel, and optionally synchronizes until control can be passed back to host.
    */
    void launch(
        dim3 grid,
        dim3 block,
        size_t shared_mem_size,
        void* cuda_stream,
        std::vector<void*> args,
        bool synchronize = true
    ) {

        if (!compiled) {
            this->compileKernel(args);
        }

        auto& driver = CUDADriver::instance();

        CUcontext currentContext = nullptr;
        // Get current context
        CUresult result = driver.cuCtxGetCurrent(&currentContext);

        if (result != CUDA_SUCCESS || !currentContext) {
            throw std::runtime_error("CachedKernel::launch error getting current context.");
        }

        if (currentContext != context) {
            CUDADRIVER_SAFE_CALL(driver.cuCtxSetCurrent(context));
        }

        this->checkAndAdjustSharedMem(shared_mem_size);

        cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

        CUDADRIVER_SAFE_CALL(driver.cuLaunchKernel(
            function,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem_size,
            cstream,
            args.data(),
            0
        ));

        if (synchronize)
            CUDADRIVER_SAFE_CALL(driver.cuCtxSynchronize());

        if (currentContext != context) {
            CUDADRIVER_SAFE_CALL(driver.cuCtxSetCurrent(currentContext));
        }
    }

  private:
    /*
    The default shared memory space on most recent NVIDIA cards is defaulted
    49152 bytes. This method
    attempts to adjust the shared memory to fit the requested configuration if
    the kernel launch parameters exceeds the default 49152 bytes.
    */
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        auto& driver = CUDADriver::instance();
        if (current_smem_size == 0) {
            CUdevice cuDevice;
            CUresult res = driver.cuCtxGetDevice(&cuDevice);

            CUDADRIVER_SAFE_CALL(driver.cuDeviceGetAttribute(
                &max_smem_size_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice
            ));

            int reserved_smem_per_block = 0;

            CUDADRIVER_SAFE_CALL(driver.cuDeviceGetAttribute(
                &reserved_smem_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            int curr_max_smem_per_block = 0;

            CUDADRIVER_SAFE_CALL(driver.cuDeviceGetAttribute(
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
                CUDADRIVER_SAFE_CALL(driver.cuFuncSetAttribute(
                    function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, query_shared_mem_size
                ));
                current_smem_size = query_shared_mem_size;
            }
        }
    }

    /*
        Compiles the kernel "kernel_name" located in source file "kernel_code", which additional
        parameters "options" passed to nvrtc. Will auto-detect the compute capability of the
       available card. args for the launch need to be queried as we need to grab the CUcontext in
       which these ptrs exist.
        */
    void compileKernel(std::vector<void*>& kernel_args) {

        this->initCudaDriver();

        auto& driver = CUDADriver::instance();
        auto& nvrtc = NVRTC::instance();

        CUcontext currentContext = nullptr;

        for (size_t ptr_id = 0; ptr_id < kernel_args.size(); ptr_id++) {
            unsigned int memtype = 0;
            CUdeviceptr device_ptr = *reinterpret_cast<CUdeviceptr*>(kernel_args[ptr_id]);

            CUresult res =
                driver.cuPointerGetAttribute(&memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr);

            if (res == CUDA_SUCCESS && memtype == CU_MEMORYTYPE_DEVICE) {
                CUDADRIVER_SAFE_CALL(driver.cuPointerGetAttribute(
                    &currentContext, CU_POINTER_ATTRIBUTE_CONTEXT, device_ptr
                ));

                if (currentContext) {
                    break;
                }
            }
        }

        CUcontext query = nullptr;
        CUDADRIVER_SAFE_CALL(driver.cuCtxGetCurrent(&query));

        if (query != currentContext) {
            CUDADRIVER_SAFE_CALL(driver.cuCtxSetCurrent(currentContext));
        }

        CUdevice cuDevice;
        CUDADRIVER_SAFE_CALL(driver.cuCtxGetDevice(&cuDevice));

        nvrtcProgram prog;

        NVRTC_SAFE_CALL(nvrtc.nvrtcCreateProgram(
            &prog, this->kernel_code.c_str(), this->source_name.c_str(), 0, nullptr, nullptr
        ));

        NVRTC_SAFE_CALL(nvrtc.nvrtcAddNameExpression(prog, this->kernel_name.c_str()));

        std::vector<const char*> c_options;
        c_options.reserve(this->options.size());
        for (const auto& option : this->options) {
            c_options.push_back(option.c_str());
        }

        int major = 0;
        int minor = 0;
        CUDADRIVER_SAFE_CALL(driver.cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
        ));
        CUDADRIVER_SAFE_CALL(driver.cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
        ));
        int arch = major * 10 + minor;
        std::string smbuf = "--gpu-architecture=sm_" + std::to_string(arch);
        c_options.push_back(smbuf.c_str());

        nvrtcResult compileResult =
            nvrtc.nvrtcCompileProgram(prog, c_options.size(), c_options.data());
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtc.nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            NVRTC_SAFE_CALL(nvrtc.nvrtcGetProgramLog(prog, &log[0]));
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program:\n" + log
            );
        }

        // Get PTX code
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtc.nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptxCode(ptxSize);
        NVRTC_SAFE_CALL(nvrtc.nvrtcGetPTX(prog, ptxCode.data()));

        CUmodule module;

        CUresult cuResult = driver.cuModuleLoadDataEx(&module, ptxCode.data(), 0, 0, 0);

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to load PTX code into CUDA "
                "module "
                "(error code: " +
                std::to_string(cuResult) + ")"
            );
        }

        const char* lowered_name;
        NVRTC_SAFE_CALL(nvrtc.nvrtcGetLoweredName(prog, this->kernel_name.c_str(), &lowered_name));
        CUfunction kernel;
        CUDADRIVER_SAFE_CALL(driver.cuModuleGetFunction(&kernel, module, lowered_name));

        this->set(module, kernel, currentContext);
        this->compiled = true;

        NVRTC_SAFE_CALL(nvrtc.nvrtcDestroyProgram(&prog));
    }

    void initCudaDriver() {

        auto& driver = CUDADriver::instance();

        int deviceCount = 0;
        // Check if CUDA has already been initialized
        CUresult res = driver.cuDeviceGetCount(&deviceCount);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = driver.cuInit(0);
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "KernelFactory::initCudaDriver: Failed to initialize CUDA driver."
                );
                return;
            }
        }
    }

    int current_smem_size = 0;
    int max_smem_size_optin = 0;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
    bool compiled = false;

    std::string kernel_name;
    std::string kernel_code;
    std::string source_name;
    std::vector<std::string> options;
};

/*
Factory class to create and store compiled cuda kernels for caching as a simple name-based hashmap.
ALlows both compi.ing from a source file, or for compiling from a variable containing CUDA code.
*/
class KernelFactory {

  public:
    static KernelFactory& instance() {
        static KernelFactory instance;
        return instance;
    }

    void cacheKernel(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        kernel_cache[kernel_name] =
            std::make_unique<CachedKernel>(kernel_name, source_path, source_name, options);
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

    /*
    Tries to retrieve the kernel "kernel_name". If not found, instantiate it and save to cache.
    */
    CachedKernel* createFromSource(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        if (!this->hasKernel(kernel_name)) {
            std::string kernel_code = load_cuda_source(source_path);
            this->cacheKernel(kernel_name, kernel_code, source_name, options);
        }
        return this->getKernel(kernel_name);
    }

    /*
    Tries to retrieve the kernel "kernel_name". If not found, instantiate it and save to cache.
    */
    CachedKernel* create(
        const std::string& kernel_name,
        const std::string& source_variable,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        if (!this->hasKernel(kernel_name)) {
            this->cacheKernel(kernel_name, source_variable, source_name, options);
        }

        return this->getKernel(kernel_name);
    }

  private:
    KernelFactory() {}
    std::unordered_map<std::string, std::unique_ptr<CachedKernel>> kernel_cache;

    KernelFactory(const KernelFactory&) = delete;
    KernelFactory& operator=(const KernelFactory&) = delete;
};

#endif