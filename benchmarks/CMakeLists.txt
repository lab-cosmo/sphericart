add_executable(benchmark_cpp cpp/benchmark.cpp)
target_link_libraries(benchmark_cpp sphericart)
target_compile_features(benchmark_cpp PRIVATE cxx_std_17)

add_test(
    NAME benchmark_cpp
    COMMAND ./benchmark_cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
