cmake_minimum_required(VERSION 3.10)

project(sphericart_torch CXX)

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

option(SPHERICART_TORCH_BUILD_FOR_PYTHON "Are we building sphericart_torch for usage from Python?" OFF)
mark_as_advanced(SPHERICART_TORCH_BUILD_FOR_PYTHON)

if (SPHERICART_TORCH_BUILD_FOR_PYTHON)
    if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(FATAL_ERROR "SPHERICART_TORCH_BUILD_FOR_PYTHON can only be set when this project is the root project")
    endif()

    # prevent recursive build: we are including core/CMakeLists.txt, which can
    # include this file (in core/torch/CMakeLists.txt) if SPHERICART_BUILD_TORCH=ON.
    set(SPHERICART_BUILD_TORCH OFF CACHE BOOL "" FORCE)
    mark_as_advanced(SPHERICART_BUILD_TORCH)
    add_subdirectory(core)

    # add path to the cmake configuration of the version of libtorch used
    # by the Python torch module. PYTHON_EXECUTABLE is provided by skbuild
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
        OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
        ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
    )

    if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
        message(FATAL_ERROR "failed to find your pytorch installation\n${TORCH_CMAKE_PATH_ERROR}")
    endif()

    string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")
endif()

find_package(Torch 1.13 REQUIRED)

add_library(sphericart_torch SHARED
    "include/sphericart/torch.hpp"
    "include/sphericart/autograd.hpp"
    "src/torch.cpp"
    "src/autograd.cpp"
)

target_link_libraries(sphericart_torch PUBLIC torch sphericart)
target_compile_features(sphericart_torch PUBLIC cxx_std_17)
target_include_directories(sphericart_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS sphericart_torch
    LIBRARY DESTINATION "lib"
)
