// DynamicCUDAHeader.hpp
#ifndef DYNAMIC_CUDA_HEADER_HPP
#define DYNAMIC_CUDA_HEADER_HPP

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <any>

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::cerr << "\nerror: " #x " failed with error "                                      \
                      << NVRTC::instance().nvrtcGetErrorString(result) << '\n'                     \
                      << "File: " << __FILE__ << '\n'                                              \
                      << "Line: " << __LINE__ << '\n';                                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUDADRIVER_SAFE_CALL(x)                                                                    \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            CUDADriver::instance().cuGetErrorName(result, &msg);                                   \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n'                       \
                      << "File: " << __FILE__ << '\n'                                              \
                      << "Line: " << __LINE__ << '\n';                                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUDART_SAFE_CALL(call)                                                                     \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << CUDART::instance().cudaGetErrorString(cudaStatus) << '\n'                 \
                      << "File: " << __FILE__ << '\n'                                              \
                      << "Line: " << __LINE__ << '\n';                                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Define a template to dynamically load symbols
template <typename FuncType> FuncType load(void* handle, const char* functionName) {
    auto func = reinterpret_cast<FuncType>(dlsym(handle, functionName));
    if (!func) {
        throw std::runtime_error(std::string("Failed to load function: ") + functionName);
    }
    return func;
}

/*
Possible idea is to provide the ability for end-users to dynamically load symbols that aren't
contained in their respective classes here. Might use this class later, not sure.

Example

CUDADriver& driver = CUDADriver::instance();

// Load cuInit
driver.loadSymbol<CUresult(unsigned int)>("cuInit");

// Retrieve the cuInit function as std::function
std::function<CUresult(unsigned int)> cuInit = driver.getFunction<CUresult(unsigned int)>("cuInit");

// Call the function
CUresult result = cuInit(0);
*/
class DynamicLoader {

  public:
    // Load a CUDA function symbol dynamically
    template <typename FuncType> void loadSymbol(const std::string& symbolName) {
        if (symbolRegistry.find(symbolName) == symbolRegistry.end()) {

            void* symbolPtr = dlsym(handle, symbolName.c_str());
            if (!symbolPtr) {
                throw std::runtime_error("Failed to load symbol: " + symbolName);
            }

            symbolRegistry[symbolName] =
                std::function<FuncType>(reinterpret_cast<FuncType*>(symbolPtr));
        } else {
            std::cerr << "Symbol " << symbolName << " already loaded." << std::endl;
        }
    }

    // Retrieve a function pointer by name
    template <typename FuncType>
    std::function<FuncType> getFunction(const std::string& symbolName) {
        auto it = symbolRegistry.find(symbolName);
        if (it != symbolRegistry.end()) {
            return std::any_cast<std::function<FuncType>>(it->second);
        } else {
            throw std::runtime_error("Symbol not loaded: " + symbolName);
        }
    }

  private:
    std::unordered_map<std::string, std::any> symbolRegistry;
    void* handle;
};

class CUDART {
  public:
    static CUDART& instance() {
        static CUDART instance;
        return instance;
    }

    using cudaGetDeviceCount_t = cudaError_t (*)(int*);
    using cudaGetDevice_t = cudaError_t (*)(int*);
    using cudaSetDevice_t = cudaError_t (*)(int);
    using cudaMalloc_t = cudaError_t (*)(void**, size_t);
    using cudaMemcpy_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
    using cudaGetErrorName_t = const char* (*)(cudaError_t);
    using cudaGetErrorString_t = const char* (*)(cudaError_t);
    using cudaDeviceSynchronize_t = cudaError_t (*)(void);
    using cudaPointerGetAttributes_t = cudaError_t (*)(cudaPointerAttributes*, const void*);
    using cudaFree_t = cudaError_t (*)(void*);
    // using cudaInitDevice_t = cudaError_t (*)(int, unsigned int, unsigned int);

    cudaGetDeviceCount_t cudaGetDeviceCount;
    cudaGetDevice_t cudaGetDevice;
    cudaSetDevice_t cudaSetDevice;
    cudaMalloc_t cudaMalloc;
    cudaMemcpy_t cudaMemcpy;
    cudaGetErrorName_t cudaGetErrorName;
    cudaGetErrorString_t cudaGetErrorString;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaPointerGetAttributes_t cudaPointerGetAttributes;
    cudaFree_t cudaFree;
    // cudaInitDevice_t cudaInitDevice;

  private:
    CUDART() {
        cudartHandle = dlopen("libcudart.so", RTLD_NOW);

        if (!cudartHandle) {
            throw std::runtime_error(
                "Failed to load libcudart.so. Try running \"find /usr -name libcudart.so\" and "
                "appending the directory to your $LD_LIBRARY_PATH environment variable."
            );
        }
        // load cudart function pointers using template
        cudaGetDeviceCount = load<cudaGetDeviceCount_t>(cudartHandle, "cudaGetDeviceCount");
        cudaGetDevice = load<cudaGetDevice_t>(cudartHandle, "cudaGetDevice");
        cudaSetDevice = load<cudaSetDevice_t>(cudartHandle, "cudaSetDevice");
        cudaMalloc = load<cudaMalloc_t>(cudartHandle, "cudaMalloc");
        cudaMemcpy = load<cudaMemcpy_t>(cudartHandle, "cudaMemcpy");
        cudaGetErrorName = load<cudaGetErrorName_t>(cudartHandle, "cudaGetErrorName");
        cudaGetErrorString = load<cudaGetErrorString_t>(cudartHandle, "cudaGetErrorString");
        cudaDeviceSynchronize = load<cudaDeviceSynchronize_t>(cudartHandle, "cudaDeviceSynchronize");
        cudaPointerGetAttributes =
            load<cudaPointerGetAttributes_t>(cudartHandle, "cudaPointerGetAttributes");
        cudaFree = load<cudaFree_t>(cudartHandle, "cudaFree");
        // cudaInitDevice = load<cudaInitDevice_t>(cudartHandle, "cudaInitDevice");
    }

    ~CUDART() {
        if (cudartHandle) {
            dlclose(cudartHandle);
        }
    }

    // Prevent copying
    CUDART(const CUDART&) = delete;
    CUDART& operator=(const CUDART&) = delete;

    void* cudartHandle = nullptr;
};

class CUDADriver {
  public:
    static CUDADriver& instance() {
        static CUDADriver instance;
        return instance;
    }

    using cuInit_t = CUresult (*)(unsigned int);
    using cuDeviceGetCount_t = CUresult (*)(int*);
    using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
    using cuDevicePrimaryCtxRelease_t = CUresult (*)(CUdevice);
    using cuCtxCreate_t = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
    using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
    using cuModuleLoadDataEx_t = CUresult (*)(CUmodule*, const void*, unsigned int, int*, int*);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
    using cuFuncSetAttribute_t = CUresult (*)(CUfunction, CUfunction_attribute, int);
    using cuFuncGetAttribute_t = CUresult (*)(int*, CUfunction_attribute, CUfunction);
    using cuCtxGetDevice_t = CUresult (*)(CUdevice*);
    using cuDeviceGetAttribute_t = CUresult (*)(int*, CUdevice_attribute, CUdevice);
    using cuDeviceGetName_t = CUresult (*)(char*, int, CUdevice);
    using cuDeviceTotalMem_t = CUresult (*)(size_t*, CUdevice);
    using cuLaunchKernel_t =
        CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, size_t, CUstream, void**, void*);
    using cuStreamCreate_t = CUresult (*)(CUstream*, unsigned int);
    using cuStreamDestroy_t = CUresult (*)(CUstream);
    using cuCtxSynchronize_t = CUresult (*)(void);
    using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
    using cuCtxPushCurrent_t = CUresult (*)(CUcontext);
    using cuPointerGetAttribute_t = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);

    cuInit_t cuInit;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuCtxCreate_t cuCtxCreate;
    cuCtxDestroy_t cuCtxDestroy;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain;
    cuDevicePrimaryCtxRelease_t cuDevicePrimaryCtxRelease;
    cuCtxGetCurrent_t cuCtxGetCurrent;
    cuCtxSetCurrent_t cuCtxSetCurrent;
    cuModuleLoadDataEx_t cuModuleLoadDataEx;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuFuncSetAttribute_t cuFuncSetAttribute;
    cuFuncGetAttribute_t cuFuncGetAttribute;
    cuCtxGetDevice_t cuCtxGetDevice;
    cuDeviceGetAttribute_t cuDeviceGetAttribute;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceTotalMem_t cuDeviceTotalMem;
    cuLaunchKernel_t cuLaunchKernel;
    cuStreamCreate_t cuStreamCreate;
    cuStreamDestroy_t cuStreamDestroy;
    cuGetErrorName_t cuGetErrorName;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuCtxPushCurrent_t cuCtxPushCurrent;
    cuPointerGetAttribute_t cuPointerGetAttribute;

  private:
    CUDADriver() {
        // Load CUDA driver, cuda runtime and NVRTC libraries
        cudaHandle = dlopen("libcuda.so", RTLD_NOW);

        if (!cudaHandle) {
            throw std::runtime_error(
                "Failed to load libcuda.so. Try running \"find /usr -name libcuda.so\" and "
                "appending the directory to your $LD_LIBRARY_PATH environment variable."
            );
        }

        // Load CUDA driver function pointers using template
        cuInit = load<cuInit_t>(cudaHandle, "cuInit");
        cuDeviceGetCount = load<cuDeviceGetCount_t>(cudaHandle, "cuDeviceGetCount");
        cuCtxCreate = load<cuCtxCreate_t>(cudaHandle, "cuCtxCreate");
        cuCtxDestroy = load<cuCtxDestroy_t>(cudaHandle, "cuCtxDestroy");
        cuDevicePrimaryCtxRetain =
            load<cuDevicePrimaryCtxRetain_t>(cudaHandle, "cuDevicePrimaryCtxRetain");
        cuDevicePrimaryCtxRelease =
            load<cuDevicePrimaryCtxRelease_t>(cudaHandle, "cuDevicePrimaryCtxRelease");
        cuCtxGetCurrent = load<cuCtxGetCurrent_t>(cudaHandle, "cuCtxGetCurrent");
        cuCtxSetCurrent = load<cuCtxSetCurrent_t>(cudaHandle, "cuCtxSetCurrent");
        cuModuleLoadDataEx = load<cuModuleLoadDataEx_t>(cudaHandle, "cuModuleLoadDataEx");
        cuModuleGetFunction = load<cuModuleGetFunction_t>(cudaHandle, "cuModuleGetFunction");
        cuFuncSetAttribute = load<cuFuncSetAttribute_t>(cudaHandle, "cuFuncSetAttribute");
        cuFuncGetAttribute = load<cuFuncGetAttribute_t>(cudaHandle, "cuFuncGetAttribute");
        cuCtxGetDevice = load<cuCtxGetDevice_t>(cudaHandle, "cuCtxGetDevice");
        cuDeviceGetAttribute = load<cuDeviceGetAttribute_t>(cudaHandle, "cuDeviceGetAttribute");
        cuDeviceGetName = load<cuDeviceGetName_t>(cudaHandle, "cuDeviceGetName");
        cuDeviceTotalMem = load<cuDeviceTotalMem_t>(cudaHandle, "cuDeviceTotalMem");
        cuLaunchKernel = load<cuLaunchKernel_t>(cudaHandle, "cuLaunchKernel");
        cuStreamCreate = load<cuStreamCreate_t>(cudaHandle, "cuStreamCreate");
        cuStreamDestroy = load<cuStreamDestroy_t>(cudaHandle, "cuStreamDestroy");
        cuCtxSynchronize = load<cuCtxSynchronize_t>(cudaHandle, "cuCtxSynchronize");
        cuGetErrorName = load<cuGetErrorName_t>(cudaHandle, "cuGetErrorName");
        cuCtxPushCurrent = load<cuCtxPushCurrent_t>(cudaHandle, "cuCtxPushCurrent");
        cuPointerGetAttribute = load<cuPointerGetAttribute_t>(cudaHandle, "cuPointerGetAttribute");
    }

    ~CUDADriver() {
        if (cudaHandle) {
            dlclose(cudaHandle);
        }
    }

    // Prevent copying
    CUDADriver(const CUDADriver&) = delete;
    CUDADriver& operator=(const CUDADriver&) = delete;

    void* cudaHandle = nullptr;
};

class NVRTC {
  public:
    static NVRTC& instance() {
        static NVRTC instance;
        return instance;
    }

    using nvrtcCreateProgram_t =
        nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char*[], const char*[]);
    using nvrtcCompileProgram_t = nvrtcResult (*)(nvrtcProgram, int, const char*[]);
    using nvrtcGetPTX_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetPTXSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcGetProgramLog_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetProgramLogSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcAddNameExpression_t = nvrtcResult (*)(nvrtcProgram, const char* const);
    using nvrtcGetLoweredName_t = nvrtcResult (*)(nvrtcProgram, const char*, const char**);
    using nvrtcDestroyProgram_t = nvrtcResult (*)(nvrtcProgram*);
    using nvrtcGetErrorString_t = const char* (*)(nvrtcResult);

    nvrtcCreateProgram_t nvrtcCreateProgram;
    nvrtcCompileProgram_t nvrtcCompileProgram;
    nvrtcGetPTX_t nvrtcGetPTX;
    nvrtcGetPTXSize_t nvrtcGetPTXSize;
    nvrtcGetProgramLog_t nvrtcGetProgramLog;
    nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize;
    nvrtcGetLoweredName_t nvrtcGetLoweredName;
    nvrtcAddNameExpression_t nvrtcAddNameExpression;
    nvrtcDestroyProgram_t nvrtcDestroyProgram;
    nvrtcGetErrorString_t nvrtcGetErrorString;

  private:
    NVRTC() {
        nvrtcHandle = dlopen("libnvrtc.so", RTLD_NOW);

        if (!nvrtcHandle) {
            throw std::runtime_error(
                "Failed to load libnvrtc.so. Try running \"find /usr -name libnvrtc.so\" and "
                "appending the directory to your $LD_LIBRARY_PATH environment variable."
            );
        }

        // Load NVRTC function pointers using template
        nvrtcCreateProgram = load<nvrtcCreateProgram_t>(nvrtcHandle, "nvrtcCreateProgram");
        nvrtcCompileProgram = load<nvrtcCompileProgram_t>(nvrtcHandle, "nvrtcCompileProgram");
        nvrtcGetPTX = load<nvrtcGetPTX_t>(nvrtcHandle, "nvrtcGetPTX");
        nvrtcGetPTXSize = load<nvrtcGetPTXSize_t>(nvrtcHandle, "nvrtcGetPTXSize");
        nvrtcGetProgramLog = load<nvrtcGetProgramLog_t>(nvrtcHandle, "nvrtcGetProgramLog");
        nvrtcGetProgramLogSize =
            load<nvrtcGetProgramLogSize_t>(nvrtcHandle, "nvrtcGetProgramLogSize");
        nvrtcGetLoweredName = load<nvrtcGetLoweredName_t>(nvrtcHandle, "nvrtcGetLoweredName");
        nvrtcAddNameExpression =
            load<nvrtcAddNameExpression_t>(nvrtcHandle, "nvrtcAddNameExpression");
        nvrtcDestroyProgram = load<nvrtcDestroyProgram_t>(nvrtcHandle, "nvrtcDestroyProgram");
        nvrtcGetErrorString = load<nvrtcGetErrorString_t>(nvrtcHandle, "nvrtcGetErrorString");
    }

    ~NVRTC() {
        if (nvrtcHandle) {
            dlclose(nvrtcHandle);
        }
    }

    // Prevent copying
    NVRTC(const NVRTC&) = delete;
    NVRTC& operator=(const NVRTC&) = delete;

    void* nvrtcHandle = nullptr;
};

#endif // DYNAMIC_CUDA_HEADER_HPP