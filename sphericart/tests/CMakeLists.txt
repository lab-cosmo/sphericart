add_executable(test_hardcoding test_hardcoding.cpp)
target_link_libraries(test_hardcoding sphericart)
target_compile_features(test_hardcoding PRIVATE cxx_std_17)

add_executable(test_samples test_samples.cpp)
target_link_libraries(test_samples sphericart)
target_compile_features(test_samples PRIVATE cxx_std_17)

add_test(NAME test_hardcoding COMMAND ./test_hardcoding)
add_test(NAME test_samples COMMAND ./test_samples)
