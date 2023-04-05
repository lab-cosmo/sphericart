add_executable(test_hardcoding test_hardcoding.cpp)
target_link_libraries(test_hardcoding sphericart)
target_compile_features(test_hardcoding PRIVATE cxx_std_17)

add_test(NAME test_hardcoding COMMAND ./test_hardcoding)
