cmake_minimum_required(VERSION 3.0)
message(CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -Wall -fopenmp")
set(CMAKE_C_FLAGS_SINGLETHREAD "-g -O2 -march=native -Wall")
project(
    sphericart
    VERSION 0.1
    LANGUAGES C)
add_library(sphericart SHARED sphericart.c)
install(TARGETS sphericart 
    DESTINATION "${CMAKE_INSTALL_PREFIX}")
