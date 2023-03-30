add_executable(example_cpp example.cpp)
target_link_libraries(example_cpp sphericart)
add_test(NAME example_cpp COMMAND ./example_cpp)

add_executable(example_c example.c)
target_link_libraries(example_c sphericart)
add_test(NAME example_c COMMAND ./example_c)
