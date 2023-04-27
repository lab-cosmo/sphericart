add_executable(example_cpp cpp/example.cpp)
target_link_libraries(example_cpp sphericart)
target_compile_features(example_cpp PRIVATE cxx_std_14)
add_test(
    NAME example_cpp
    COMMAND ./example_cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(example_c c/example.c)
target_link_libraries(example_c sphericart)
add_test(
    NAME example_c
    COMMAND example_c
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
