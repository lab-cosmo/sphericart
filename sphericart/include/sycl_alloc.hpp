// Authors: abagusetty@github and alvarovm@github , Argonne UChicago LLC.
#include "sycl_device.hpp"

// Function to check SYCL errors
template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "SYCL error at " << file << ":" << line << " code=" << result << " \"" << func
                  << "\" \n";
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) (val)

#define MALLOC(type, var, size)                                                                    \
    type* var = ::sycl::malloc_device<type>(size, *(sycl_get_queue()));                            \
    if (var == nullptr) {                                                                          \
        std::cerr << "Memory allocation failed for " #var " at " __FILE__ ":" << __LINE__          \
                  << std::endl;                                                                    \
        std::exit(EXIT_FAILURE);                                                                   \
    }

#define FREE(var) ::sycl::free(var, *(sycl_get_queue()))

#define MEMSET(addr, val, size)                                                                      \
    {                                                                                                \
        sycl_get_queue()->submit([&](::sycl::handler& cgh) { cgh.memset(addr, val, size); }).wait(); \
    }

#define DEVICE_INIT(type, dst, src, size)                                                          \
    MALLOC(type, dst, size);                                                                       \
    {                                                                                              \
        sycl_get_queue()                                                                           \
            ->submit([&](::sycl::handler& cgh) { cgh.memcpy(dst, src, sizeof(type) * (size)); })   \
            .wait();                                                                               \
    }

#define DEVICE_GET(type, dst, src, size)                                                           \
    {                                                                                              \
        sycl_get_queue()                                                                           \
            ->submit([&](::sycl::handler& cgh) { cgh.memcpy(dst, src, sizeof(type) * (size)); })   \
            .wait();                                                                               \
    }
