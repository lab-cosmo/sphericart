add_executable(benchmark_cpp cpp/benchmark.cpp)
target_link_libraries(benchmark_cpp sphericart)
add_test(benchmark benchmark_cpp COMMAND ./benchmark_cpp)
target_compile_features(benchmark_cpp PRIVATE cxx_std_17)
