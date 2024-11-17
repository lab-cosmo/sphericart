cmake_minimum_required(VERSION 3.10)

if (POLICY CMP0076)
    #  target_sources() converts relative paths to absolute
    cmake_policy(SET CMP0076 NEW)
endif()

project(sphericart_torch CXX)

if(COMPILER_SUPPORTS_WPRAGMAS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(STATUS "Could not find a CUDA compiler")
endif()

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

option(SPHERICART_TORCH_BUILD_FOR_PYTHON "Are we building sphericart_torch for usage from Python?" OFF)
mark_as_advanced(SPHERICART_TORCH_BUILD_FOR_PYTHON)

if (SPHERICART_TORCH_BUILD_FOR_PYTHON)
    if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(FATAL_ERROR "SPHERICART_TORCH_BUILD_FOR_PYTHON can only be set when this project is the root project")
    endif()

    # prevent recursive build: we are including sphericart/CMakeLists.txt, which can
    # include this file (in sphericart/torch/CMakeLists.txt) if SPHERICART_BUILD_TORCH=ON.
    set(SPHERICART_BUILD_TORCH OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    mark_as_advanced(SPHERICART_BUILD_TORCH)
    add_subdirectory(sphericart EXCLUDE_FROM_ALL)

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

    # ============================= Handle CUDNN ============================= #
    # The FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
    # fixed version in `cmake/FindCUDNN.cmake`, and we set the right variables
    # for it below to point the code to the right `libcudnn.do`/`cudnn.h`
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

    # First try using the `nvidia.cudnn` package (dependency of torch on PyPI)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))"
        RESULT_VARIABLE CUDNN_CMAKE_PATH_RESULT
        OUTPUT_VARIABLE CUDNN_CMAKE_PATH_OUTPUT
        ERROR_VARIABLE CUDNN_CMAKE_PATH_ERROR
    )
    if (${CUDNN_CMAKE_PATH_RESULT} EQUAL 0)
        string(STRIP ${CUDNN_CMAKE_PATH_OUTPUT} CUDNN_CMAKE_PATH_OUTPUT)
        set(CUDNN_ROOT ${CUDNN_CMAKE_PATH_OUTPUT})
    else()
        # Otherwise try to find CuDNN inside PyTorch itself
        set(CUDNN_ROOT ${TORCH_CMAKE_PATH_OUTPUT}/../..)

        if (NOT EXISTS ${CUDNN_ROOT}/include/cudnn_version.h)
            # HACK: create a minimal cudnn_version.h (with a made-up version),
            # because it is not bundled together with the CuDNN shared library
            # in PyTorch: https://github.com/pytorch/pytorch/issues/47743
            file(WRITE ${CUDNN_ROOT}/include/cudnn_version.h "#define CUDNN_MAJOR 8\n#define CUDNN_MINOR 5\n#define CUDNN_PATCHLEVEL 0\n")
        endif()
    endif()

    set(CUDNN_INCLUDE_DIR ${CUDNN_ROOT}/include CACHE PATH "" FORCE)
    set(CUDNN_LIBRARY ${CUDNN_ROOT}/lib CACHE PATH "" FORCE)
    unset(CUDNN_ROOT)

    mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
else()
    if (NOT TARGET sphericart)
        message(FATAL_ERROR "missing sphericart target, you should build sphericart_torch as a sub-project of sphericart")
    endif()
endif()

find_package(Torch 1.13 REQUIRED)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_build_torch_version.py "BUILD_TORCH_VERSION = '${Torch_VERSION}'")

message(STATUS "DEBUG PyTorch Version: ${Torch_VERSION}")

add_library(sphericart_torch SHARED
    "include/sphericart/torch_cuda_wrapper.hpp"
    "include/sphericart/torch.hpp"
    "include/sphericart/autograd.hpp"
    "src/autograd.cpp"
    "src/torch.cpp"
)

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    target_sources(sphericart_torch PUBLIC "src/torch_cuda_wrapper.cpp")
else()
    target_sources(sphericart_torch PUBLIC "src/torch_cuda_wrapper_stub.cpp")
endif()

target_link_libraries(sphericart_torch PUBLIC torch sphericart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sphericart_torch PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
    target_include_directories(sphericart_torch PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(sphericart_torch PRIVATE CUDA_AVAILABLE)
endif()
target_compile_features(sphericart_torch PUBLIC cxx_std_17)

target_include_directories(sphericart_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS sphericart_torch
    LIBRARY DESTINATION "lib"
)

if (SPHERICART_TORCH_BUILD_FOR_PYTHON)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/_build_torch_version.py
        DESTINATION "."
    )
endif()
