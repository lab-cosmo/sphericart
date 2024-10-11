// DynamicCUDAHeader.hpp
#ifndef DYNAMIC_CUDA_HEADER_HPP
#define DYNAMIC_CUDA_HEADER_HPP

#ifdef __linux__
#include <dlfcn.h>
#else
#error "Platform not supported"
#endif
#include <stdexcept>
#include <string>
#include <functional>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <any>
#include <sstream>

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::ostringstream errorMsg;                                                           \
            errorMsg << "\nerror: " #x " failed with error " << nvrtc.nvrtcGetErrorString(result)  \
                     << '\n'                                                                       \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
        }                                                                                          \
    } while (0)

#define CUDADRIVER_SAFE_CALL(x)                                                                    \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            cudadriver.cuGetErrorName(result, &msg);                                               \
            std::ostringstream errorMsg;                                                           \
            errorMsg << "\nerror: " #x " failed with error " << (msg ? msg : "Unknown error")      \
                     << '\n'                                                                       \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
        }                                                                                          \
    } while (0)

#define CUDART_SAFE_CALL(call)                                                                     \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::ostringstream errorMsg;                                                           \
            const char* error = cudart.cudaGetErrorString(cudaStatus);                             \
            errorMsg << "\nfailed with error " << (error ? error : "Unknown error") << '\n'        \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
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

class CUDART {
  public:
    bool loaded() { return cudartHandle != nullptr; }

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

    CUDART() {
#ifdef __linux__
        cudartHandle = dlopen("libcudart.so", RTLD_NOW);
#else
#error "Platform not supported"
#endif
        if (!cudartHandle) {
            throw std::runtime_error("Failed to load libcudart.so. Make sure it is available in "
                                     "your $LD_LIBRARY_PATH environment variable.");
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
    }

    ~CUDART() {
#ifdef __linux__
        if (cudartHandle) {
            // Do not close - leak these for OS cleanup to avoid Static Initialization Order Fiasco
            dlclose(cudartHandle);
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDART(const CUDART&) = delete;
    CUDART& operator=(const CUDART&) = delete;

    void* cudartHandle = nullptr;
};

class CUDADriver {

  public:
    bool loaded() { return cudaHandle != nullptr; }

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

    CUDADriver() {
#ifdef __linux__
        cudaHandle = dlopen("libcuda.so", RTLD_NOW);
#else
#error "Platform not supported"
#endif
        if (!cudaHandle) {
            throw std::runtime_error("Failed to load libcuda.so. Make sure it is available in your "
                                     "$LD_LIBRARY_PATH environment variable.");
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
#ifdef __linux__
        if (cudaHandle) {
            // Do not close - leak these for OS cleanup to avoid Static Initialization Order Fiasco
            dlclose(cudaHandle);
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDADriver(const CUDADriver&) = delete;
    CUDADriver& operator=(const CUDADriver&) = delete;

    void* cudaHandle = nullptr;
};

class NVRTC {

  public:
    bool loaded() { return nvrtcHandle != nullptr; }

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

    NVRTC() {
#ifdef __linux__
        nvrtcHandle = dlopen("libnvrtc.so", RTLD_NOW);
#else
#error "Platform not supported"
#endif

        if (!nvrtcHandle) {
            throw std::runtime_error("Failed to load libnvrtc.so. Make sure it is available in "
                                     "your $LD_LIBRARY_PATH environment variable.");
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
#ifdef __linux__
        if (nvrtcHandle) {
            // Do not close - leak these for OS cleanup to avoid Static Initialization Order Fiasco
            dlclose(nvrtcHandle);
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    NVRTC(const NVRTC&) = delete;
    NVRTC& operator=(const NVRTC&) = delete;

    void* nvrtcHandle = nullptr;
};

/*This implements the Schawrz counter idiom to ensure propper constructor/destructor ordering*/

// Static memory buffer for each class
static std::aligned_storage<sizeof(CUDART), alignof(CUDART)>::type cudartBuffer;
static std::aligned_storage<sizeof(CUDADriver), alignof(CUDADriver)>::type cudaDriverBuffer;
static std::aligned_storage<sizeof(NVRTC), alignof(NVRTC)>::type nvrtcBuffer;

// global references
inline CUDART& cudart = reinterpret_cast<CUDART&>(cudartBuffer);
inline CUDADriver& cudadriver = reinterpret_cast<CUDADriver&>(cudaDriverBuffer);
inline NVRTC& nvrtc = reinterpret_cast<NVRTC&>(nvrtcBuffer);

static int nifty_counter = 0;

// Counter class for initializing and destroying static objects
static struct CUDAInitializer {

    CUDAInitializer() {
        if (nifty_counter++ == 0) {
            new (&cudart) CUDART();
            new (&cudadriver) CUDADriver();
            new (&nvrtc) NVRTC();
        }
    }

    ~CUDAInitializer() {
        if (--nifty_counter == 0) {
            cudart.~CUDART();
            cudadriver.~CUDADriver();
            nvrtc.~NVRTC();
        }
    }

} cudaInitializer;

#endif // DYNAMIC_CUDA_HEADER_HPP