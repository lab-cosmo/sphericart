add_executable(timings timings.cpp)
target_link_libraries(timings sphericart)

add_test(NAME timings COMMAND ./timings)