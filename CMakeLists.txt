cmake_minimum_required(VERSION 3.10)

project(sphericart CXX)

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

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

file(GLOB_RECURSE chemfiles_objects_sources src/**.cpp)
add_library(sphericart
    "src/sphericart.cpp"
    "src/sphericart-capi.cpp"
    "include/sphericart.hpp"
    "include/sphericart.h"
)

set_target_properties(sphericart PROPERTIES
    VERSION "0.1.0"
    SOVERSION "0.1"
)

set_target_properties(sphericart PROPERTIES
    # Set the version of C++ standard we need
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
)

set(CMAKE_CXX_FLAGS "-g -O2 -march=native -Wall -fopenmp")

target_include_directories(sphericart PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(sphericart PRIVATE sphericart_EXPORTS)

include(GenerateExportHeader)
generate_export_header(sphericart
    BASE_NAME SPHERICART
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/sphericart/exports.h
)

# TODO: installation from cmake
