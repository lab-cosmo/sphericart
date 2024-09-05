// DynamicCUDAHeader.hpp
#ifndef DYNAMIC_CUDA_HEADER_HPP
#define DYNAMIC_CUDA_HEADER_HPP

#include <dlfcn.h>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

// Handle dynamic loading and function retrieval
class DynamicCUDA {
  public:
    static DynamicCUDA& instance() {
        static DynamicCUDA instance;
        return instance;
    }

    // Function pointers for CUDA and NVRTC functions
    using cuInit_t = CUresult (*)(unsigned int);
    using cuDeviceGetCount_t = CUresult (*)(int*);
    using cuCtxCreate_t = CUresult (*)(CUcontext*, unsigned int, CUdevice);
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

    // Public methods to access function pointers
    cuInit_t cuInit;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuCtxCreate_t cuCtxCreate;
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

    nvrtcCreateProgram_t nvrtcCreateProgram;
    nvrtcCompileProgram_t nvrtcCompileProgram;
    nvrtcGetPTX_t nvrtcGetPTX;
    nvrtcGetPTXSize_t nvrtcGetPTXSize;
    nvrtcGetProgramLog_t nvrtcGetProgramLog;
    nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize;
    nvrtcGetLoweredName_t nvrtcGetLoweredName;
    nvrtcAddNameExpression_t nvrtcAddNameExpression;
    nvrtcDestroyProgram_t nvrtcDestroyProgram;

  private:
    DynamicCUDA() {
        // Load CUDA and NVRTC libraries
        cudaHandle = dlopen("libcuda.so", RTLD_NOW);

        if (!cudaHandle) {
            throw std::runtime_error(
                "Failed to load libcuda.so. Try running \"find /usr -name libcuda.so\" and "
                "appending the directory to your $LD_LIBRARY_PATH environment variable."
            );
        }

        nvrtcHandle = dlopen("libnvrtc.so", RTLD_NOW);

        if (!nvrtcHandle) {
            dlclose(cudaHandle);
            throw std::runtime_error(
                "Failed to load libnvrtc.so. Try running \"find /usr -name libnvrtc.so\" and "
                "appending the directory to your $LD_LIBRARY_PATH environment variable."
            );
        }

        // Load CUDA function pointers
        cuInit = (cuInit_t)dlsym(cudaHandle, "cuInit");
        cuDeviceGetCount = (cuDeviceGetCount_t)dlsym(cudaHandle, "cuDeviceGetCount");
        cuCtxCreate = (cuCtxCreate_t)dlsym(cudaHandle, "cuCtxCreate");
        cuCtxGetCurrent = (cuCtxGetCurrent_t)dlsym(cudaHandle, "cuCtxGetCurrent");
        cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cudaHandle, "cuCtxSetCurrent");
        cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym(cudaHandle, "cuModuleLoadDataEx");
        cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(cudaHandle, "cuModuleGetFunction");
        cuFuncSetAttribute = (cuFuncSetAttribute_t)dlsym(cudaHandle, "cuFuncSetAttribute");
        cuFuncGetAttribute = (cuFuncGetAttribute_t)dlsym(cudaHandle, "cuFuncGetAttribute");
        cuCtxGetDevice = (cuCtxGetDevice_t)dlsym(cudaHandle, "cuCtxGetDevice");
        cuDeviceGetAttribute = (cuDeviceGetAttribute_t)dlsym(cudaHandle, "cuDeviceGetAttribute");
        cuDeviceGetName = (cuDeviceGetName_t)dlsym(cudaHandle, "cuDeviceGetName");
        cuDeviceTotalMem = (cuDeviceTotalMem_t)dlsym(cudaHandle, "cuDeviceTotalMem");
        cuLaunchKernel = (cuLaunchKernel_t)dlsym(cudaHandle, "cuLaunchKernel");
        cuStreamCreate = (cuStreamCreate_t)dlsym(cudaHandle, "cuStreamCreate");
        cuStreamDestroy = (cuStreamDestroy_t)dlsym(cudaHandle, "cuStreamDestroy");
        cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cudaHandle, "cuCtxSynchronize");
        cuGetErrorName = (cuGetErrorName_t)dlsym(cudaHandle, "cuGetErrorName");

        // Load NVRTC function pointers
        nvrtcCreateProgram = (nvrtcCreateProgram_t)dlsym(nvrtcHandle, "nvrtcCreateProgram");
        nvrtcCompileProgram = (nvrtcCompileProgram_t)dlsym(nvrtcHandle, "nvrtcCompileProgram");
        nvrtcGetPTX = (nvrtcGetPTX_t)dlsym(nvrtcHandle, "nvrtcGetPTX");
        nvrtcGetPTXSize = (nvrtcGetPTXSize_t)dlsym(nvrtcHandle, "nvrtcGetPTXSize");
        nvrtcGetProgramLog = (nvrtcGetProgramLog_t)dlsym(nvrtcHandle, "nvrtcGetProgramLog");
        nvrtcGetProgramLogSize =
            (nvrtcGetProgramLogSize_t)dlsym(nvrtcHandle, "nvrtcGetProgramLogSize");
        nvrtcGetLoweredName = (nvrtcGetLoweredName_t)dlsym(nvrtcHandle, "nvrtcGetLoweredName");
        nvrtcAddNameExpression =
            (nvrtcAddNameExpression_t)dlsym(nvrtcHandle, "nvrtcAddNameExpression");
        nvrtcDestroyProgram = (nvrtcDestroyProgram_t)dlsym(nvrtcHandle, "nvrtcDestroyProgram");

        // Check for missing symbols
        if (!cuInit || !cuDeviceGetCount || !cuCtxCreate || !cuCtxGetCurrent || !cuCtxSetCurrent ||
            !cuModuleLoadDataEx || !cuModuleGetFunction || !cuFuncSetAttribute ||
            !cuFuncGetAttribute || !cuCtxGetDevice || !cuDeviceGetAttribute || !cuDeviceGetName ||
            !cuDeviceTotalMem || !cuLaunchKernel || !cuStreamCreate || !cuStreamDestroy ||
            !nvrtcCreateProgram || !nvrtcCompileProgram || !nvrtcGetPTX || !nvrtcGetProgramLog ||
            !nvrtcGetLoweredName || !nvrtcDestroyProgram) {
            dlclose(cudaHandle);
            dlclose(nvrtcHandle);
            throw std::runtime_error("Failed to load one or more CUDA/NVRTC functions. Post an "
                                     "issue here: https://github.com/lab-cosmo/sphericart/issues.");
        }
    }

    ~DynamicCUDA() {
        if (cudaHandle) {
            dlclose(cudaHandle);
        }
        if (nvrtcHandle) {
            dlclose(nvrtcHandle);
        }
    }

    // Prevent copying
    DynamicCUDA(const DynamicCUDA&) = delete;
    DynamicCUDA& operator=(const DynamicCUDA&) = delete;

    void* cudaHandle = nullptr;
    void* nvrtcHandle = nullptr;
};

#endif // DYNAMIC_CUDA_HEADER_HPP