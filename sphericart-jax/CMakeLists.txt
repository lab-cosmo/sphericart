cmake_minimum_required(VERSION 3.27)

if (POLICY CMP0076)
    #  target_sources() converts relative paths to absolute
    cmake_policy(SET CMP0076 NEW)
endif()

project(sphericart_jax CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
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

# Locate Python (used to find the jaxlib include directory)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find the XLA/FFI header-only library shipped inside jaxlib
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import pathlib, jaxlib; print((pathlib.Path(jaxlib.__file__).parent / 'include').resolve())"
    OUTPUT_VARIABLE XLA_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXISTS "${XLA_INCLUDE_DIR}")
    message(FATAL_ERROR "Could not find jaxlib include directory: ${XLA_INCLUDE_DIR}")
endif()

message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# Build sphericart (static) and then link it into our shared libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(sphericart EXCLUDE_FROM_ALL)

# --- CPU shared library ---
add_library(sphericart_jax_cpu SHARED ${CMAKE_CURRENT_LIST_DIR}/src/sphericart_jax_cpu.cpp)
target_link_libraries(sphericart_jax_cpu PRIVATE sphericart)
target_compile_features(sphericart_jax_cpu PRIVATE cxx_std_17)
target_include_directories(sphericart_jax_cpu PRIVATE
    ${XLA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
install(TARGETS sphericart_jax_cpu
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
)

# --- CUDA shared library ---
if(CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    add_library(sphericart_jax_cuda SHARED ${CMAKE_CURRENT_LIST_DIR}/src/sphericart_jax_cuda.cpp)
    target_link_libraries(sphericart_jax_cuda PRIVATE sphericart)
    target_compile_features(sphericart_jax_cuda PRIVATE cxx_std_17)
    target_include_directories(sphericart_jax_cuda PRIVATE
        ${XLA_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    install(TARGETS sphericart_jax_cuda
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
    )
endif()
