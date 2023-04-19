cmake_minimum_required(VERSION 3.10)
project(sphericart_meta)
enable_testing()

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'relwithdebinfo' as none was specified.")
        set(CMAKE_BUILD_TYPE "relwithdebinfo"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

add_subdirectory(sphericart)

OPTION(SPHERICART_BUILD_TORCH "Build the torch bindings" OFF)
OPTION(SPHERICART_BUILD_EXAMPLES "Build and run examples and benchmarks for Sphericart" OFF)

if (SPHERICART_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(benchmarks)
endif()

if (SPHERICART_BUILD_TORCH)
    add_subdirectory(sphericart-torch)
endif()
