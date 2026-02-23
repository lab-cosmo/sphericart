cmake_minimum_required(VERSION 3.27)
project(sphericart_meta)
enable_testing()

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()


OPTION(SPHERICART_BUILD_TORCH "Build the torch bindings" OFF)
OPTION(SPHERICART_BUILD_EXAMPLES "Build and run examples and benchmarks for Sphericart" OFF)
OPTION(SPHERICART_PRINT_DEBUG "Enable debug print statements" OFF)

# SYCL floating-point type: "double" or "float"
set(SPHERICART_DTYPE "double" CACHE STRING "Floating-point type for SYCL backend (double or float) for testing purposes")
set_property(CACHE SPHERICART_DTYPE PROPERTY STRINGS double float)
# Define DTYPE for all targets in this project and its subdirectories
add_compile_definitions(DTYPE=${SPHERICART_DTYPE})


if (SPHERICART_PRINT_DEBUG)
    add_compile_definitions(PRINT_DEBUG)
endif()

add_subdirectory(sphericart)

if (SPHERICART_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(benchmarks)
endif()

if (SPHERICART_BUILD_TORCH)
    add_subdirectory(sphericart-torch)
endif()

