cmake_minimum_required(VERSION 3.10)

if (POLICY CMP0076)
    #  target_sources() converts relative paths to absolute
    cmake_policy(SET CMP0076 NEW)
endif()

project(sphericart_jax CXX)

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'relwithdebinfo' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "relwithdebinfo"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)


if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "SPHERICART_JAX_BUILD_FOR_PYTHON can only be set when this project is the root project")
endif()

# prevent recursive build: we are including sphericart/CMakeLists.txt, which can
# include this file (in sphericart/torch/CMakeLists.txt) if SPHERICART_BUILD_JAX=ON.
set(SPHERICART_BUILD_JAX OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
mark_as_advanced(SPHERICART_BUILD_JAX)
add_subdirectory(sphericart EXCLUDE_FROM_ALL)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/src/jax.cpp)
install(TARGETS cpu_ops DESTINATION sphericart_jax)

# add_library(sphericart_jax SHARED
#     "include/sphericart/kernel_helpers.h"
#     "include/sphericart/pybind11_kernel_helpers.h"
#     "src/jax.cpp"
# )


target_link_libraries(cpu_ops PUBLIC sphericart)
target_compile_features(cpu_ops PUBLIC cxx_std_17)

target_include_directories(cpu_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS cpu_ops
    LIBRARY DESTINATION "lib"
)

