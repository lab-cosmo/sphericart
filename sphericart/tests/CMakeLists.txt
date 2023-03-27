add_executable(timings timings.cpp)
target_link_libraries(timings sphericart)
target_compile_features(timings PRIVATE cxx_std_17)

add_test(NAME timings COMMAND ./timings)