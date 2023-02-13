cmake_minimum_required(VERSION 3.0)
project(
    sphericart
    VERSION 0.1
    LANGUAGES C)
set(CMAKE_C_FLAGS "-O3 -g -march=native -Wall -fopenmp")
add_library(sphericart SHARED sphericart.c)
install(TARGETS sphericart 
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../lib/")
