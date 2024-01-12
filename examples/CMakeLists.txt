cmake_minimum_required(VERSION 3.10)
project(sphericart_examples LANGUAGES C CXX CUDA)

add_executable(example_cpp cpp/example.cpp)
target_link_libraries(example_cpp sphericart)
add_test(NAME example_cpp COMMAND ./example_cpp)
target_compile_features(example_cpp PRIVATE cxx_std_14)

add_executable(example_c c/example.c)
target_link_libraries(example_c sphericart)
add_test(NAME example_c COMMAND ./example_c)

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)

    set(CMAKE_CUDA_ARCHITECTURES native)
    
    add_executable(example_cuda cuda/example.cu)
    target_link_libraries(example_cuda sphericart)
    add_test(NAME example_cuda COMMAND ./example_cuda)
    #target_compile_features(example_cuda PRIVATE cxx_std_14)
endif()
