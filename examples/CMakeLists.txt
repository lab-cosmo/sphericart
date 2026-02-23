cmake_minimum_required(VERSION 3.10)

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    project(sphericart_examples LANGUAGES C CXX CUDA)
else()
    project(sphericart_examples LANGUAGES C CXX)
endif()

add_executable(example_cpp cpp/example.cpp)
target_link_libraries(example_cpp sphericart)
add_test(NAME example_cpp COMMAND ./example_cpp)
target_compile_features(example_cpp PRIVATE cxx_std_14)

add_executable(example_c c/example.c)
target_link_libraries(example_c sphericart)
add_test(NAME example_c COMMAND ./example_c)

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    if(SPHERICART_ARCH_NATIVE)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_executable(example_cuda cuda/example.cu)
    target_link_libraries(example_cuda sphericart)
    add_test(NAME example_cuda COMMAND ./example_cuda)
endif()
if (SPHERICART_ENABLE_SYCL)
    # SYCL_DEVICE and DTYPE are defined in the parent CMakeLists.txt
    add_executable(example_sycl sycl/example.cpp)
    target_link_libraries(example_sycl sphericart)
    #    target_compile_definitions(example_sycl PRIVATE SYCL_DEVICE=${SPHERICART_DEVICE})
    add_test(NAME example_sycl COMMAND ./example_sycl)
endif()

find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
    # SYCL_DEVICE is defined in the parent CMakeLists.txt
    add_executable(example_kokkos kokkos/example.cpp)
    target_link_libraries(example_kokkos sphericart Kokkos::kokkos)
    #target_compile_definitions(example_kokkos PRIVATE DTYPE=${SPHERICART_DTYPE})
    add_test(NAME example_kokkos COMMAND ./example_kokkos)
endif()

