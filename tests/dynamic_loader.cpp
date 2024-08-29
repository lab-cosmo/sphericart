#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime.h>

// Define function pointers for CUDA Runtime API functions
typedef cudaError_t (*cudaSetDevice_t)(int);
typedef cudaError_t (*cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t (*cudaFree_t)(void*);
typedef cudaError_t (*cudaDeviceSynchronize_t)();
typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaGetLastError_t)();
typedef const char* (*cudaGetErrorString_t)(cudaError_t);

// Function pointers
void* cuda_handle;
cudaSetDevice_t cudaSetDevice;
cudaMalloc_t cudaMalloc;
cudaMemcpy_t cudaMemcpy;
cudaFree_t cudaFree;
cudaDeviceSynchronize_t cudaDeviceSynchronize;
cudaLaunchKernel_t cudaLaunchKernel;
cudaGetLastError_t cudaGetLastError;
cudaGetErrorString_t cudaGetErrorString;

// Function to dynamically load CUDA functions
void load_cuda_functions() {
    cuda_handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!cuda_handle) {
        std::cerr << "Failed to load CUDA library: " << dlerror() << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice = (cudaSetDevice_t)dlsym(cuda_handle, "cudaSetDevice");
    cudaMalloc = (cudaMalloc_t)dlsym(cuda_handle, "cudaMalloc");
    cudaMemcpy = (cudaMemcpy_t)dlsym(cuda_handle, "cudaMemcpy");
    cudaFree = (cudaFree_t)dlsym(cuda_handle, "cudaFree");
    cudaDeviceSynchronize = (cudaDeviceSynchronize_t)dlsym(cuda_handle, "cudaDeviceSynchronize");
    cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(cuda_handle, "cudaLaunchKernel");
    cudaGetLastError = (cudaGetLastError_t)dlsym(cuda_handle, "cudaGetLastError");
    cudaGetErrorString = (cudaGetErrorString_t)dlsym(cuda_handle, "cudaGetErrorString");

    if (!cudaSetDevice || !cudaMalloc || !cudaMemcpy || !cudaFree || !cudaDeviceSynchronize || !cudaLaunchKernel || !cudaGetLastError || !cudaGetErrorString) {
        std::cerr << "Failed to get CUDA function pointers: " << dlerror() << std::endl;
        dlclose(cuda_handle);
        exit(EXIT_FAILURE);
    }
}

// Define a simple CUDA kernel
__global__ void simpleKernel(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx;
}

int main() {
    load_cuda_functions();

    // Set the device to use (assuming device 0)
    cudaSetDevice(0);

    // Allocate memory on the device
    int* d_data;
    int* h_data = new int[256];
    cudaMalloc((void**)&d_data, sizeof(int) * 256);

    // Define grid and block dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(1);

    // Launch the kernel
    void* kernel_args[] = { &d_data };
    cudaLaunchKernel((const void*)simpleKernel, blocksPerGrid, threadsPerBlock, kernel_args, 0, 0);

    // Synchronize and check for errors
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return EXIT_FAILURE;
    }

    // Copy results from device to host
    cudaMemcpy(h_data, d_data, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_data);
    delete[] h_data;
    dlclose(cuda_handle);

    return 0;
}