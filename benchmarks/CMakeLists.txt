add_executable(benchmark_cpp cpp/benchmark.cpp)
target_link_libraries(benchmark_cpp sphericart)
add_test(benchmark benchmark_cpp COMMAND ./benchmark_cpp)
target_compile_features(benchmark_cpp PRIVATE cxx_std_17)

if (SPHERICART_ENABLE_SYCL)
    # SYCL_DEVICE and DTYPE are defined in the parent CMakeLists.txt
    add_executable(benchmark_sycl sycl/benchmark_sycl.cpp)
    target_link_libraries(benchmark_sycl sphericart)
    target_compile_features(benchmark_sycl PRIVATE cxx_std_17)
    add_test(NAME benchmark_sycl COMMAND ./benchmark_sycl)
endif()
