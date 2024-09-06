cmake_minimum_required(VERSION 3.10)

project(sphericart LANGUAGES C CXX)

file(READ ${PROJECT_SOURCE_DIR}/VERSION SPHERICART_VERSION)
string(STRIP ${SPHERICART_VERSION} SPHERICART_VERSION)
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" SPHERICART_VERSION_MAJOR "${SPHERICART_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" SPHERICART_VERSION_MINOR "${SPHERICART_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" SPHERICART_VERSION_PATCH "${SPHERICART_VERSION}")

OPTION(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

OPTION(SPHERICART_BUILD_TESTS "Build and run tests for Sphericart" OFF)
OPTION(SPHERICART_OPENMP "Try to use OpenMP when compiling Sphericart" ON)
OPTION(SPHERICART_ARCH_NATIVE "Try to use -march=native when compiling Sphericart" ON)
OPTION(SPHERICART_ENABLE_CUDA "Are we building the CUDA backend of Sphericart?" ON)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

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

set(COMMON_SOURCES
    "src/sphericart.cpp"
    "src/sphericart-capi.cpp"
    "include/sphericart.hpp"
    "include/sphericart.h"
)

# Find CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(STATUS "Could not find a CUDA compiler")
endif()

if (CMAKE_CUDA_COMPILER AND NOT SPHERICART_ENABLE_CUDA)
    message(STATUS "Found a CUDA compiler but SPHERICART_ENABLE_CUDA=OFF, set SPHERICART_ENABLE_CUDA=ON in order to compile with CUDA support.")
endif()

# Append the relevant CUDA files to sources
list(APPEND COMMON_SOURCES "include/cuda_base.hpp")
list(APPEND COMMON_SOURCES "include/sphericart_cuda.hpp")

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
list(APPEND COMMON_SOURCES "include/dynamic_cuda.hpp")
    list(APPEND COMMON_SOURCES "include/cuda_cache.hpp")
    list(APPEND COMMON_SOURCES "src/cuda_base.cu")
    list(APPEND COMMON_SOURCES "src/sphericart_cuda.cu")
    list(APPEND COMMON_SOURCES "src/sphericart_impl.cu")
    list(APPEND COMMON_SOURCES "include/sphericart_impl.cuh")
else()
    list(APPEND COMMON_SOURCES "src/cuda_stub.cpp")
    list(APPEND COMMON_SOURCES "src/sphericart_cuda_stub.cpp")
endif()

add_library(sphericart ${COMMON_SOURCES})

if (CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    #set_target_properties(sphericart PROPERTIES CUDA_NVCC_FLAGS "-lineinfo")
    target_link_libraries(sphericart INTERFACE CUDA::nvrtc)
   
    set(SPHERICART_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/sphericart/include")
    add_definitions(-DSPHERICART_INCLUDE_PATH="${SPHERICART_INCLUDE_PATH}")

    set(SPHERICART_CUDA_SRC_PATH "${CMAKE_SOURCE_DIR}/sphericart/src/sphericart_impl.cu")
    add_definitions(-DSPHERICART_CUDA_SRC_PATH="${SPHERICART_CUDA_SRC_PATH}")
endif()

set_target_properties(sphericart PROPERTIES
    VERSION ${SPHERICART_VERSION}
    SOVERSION ${SPHERICART_VERSION_MAJOR}.${SPHERICART_VERSION_MINOR}
    POSITION_INDEPENDENT_CODE ON
)

# we need to compile sphericart with C++17 for if constexpr
target_compile_features(sphericart PRIVATE cxx_std_17)

target_include_directories(sphericart PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create a header defining SPHERICART_EXPORT for exported classes/functions
set_target_properties(sphericart PROPERTIES
    # hide non-exported symbols by default
    C_VISIBILIY_PRESET hidden
    CXX_VISIBILIY_PRESET hidden
    CUDA_VISIBILIY_PRESET hidden
)

include(GenerateExportHeader)
generate_export_header(sphericart
    BASE_NAME SPHERICART
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/sphericart/exports.h
)
target_compile_definitions(sphericart PRIVATE sphericart_EXPORTS)

# Handle optimization and OpenMP flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wunknown-pragmas" COMPILER_SUPPORTS_WPRAGMAS)
if (SPHERICART_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP is enabled")
        target_link_libraries(sphericart PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "Could not find OpenMP")
        if(COMPILER_SUPPORTS_WPRAGMAS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        endif()
    endif()
else()
    if(COMPILER_SUPPORTS_WPRAGMAS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

if (SPHERICART_ARCH_NATIVE)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    # for some reason COMPILER_SUPPORTS_MARCH_NATIVE is true with Apple clang,
    # but then fails with `the clang compiler does not support '-march=native'`
    if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "march=native is enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message(STATUS "march=native is not supported by this compiler")
    endif()
endif()

# handle warning flags
check_cxx_compiler_flag("-Wall" COMPILER_SUPPORTS_WALL)
if(COMPILER_SUPPORTS_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

check_cxx_compiler_flag("-Wextra" COMPILER_SUPPORTS_WEXTRA)
if(COMPILER_SUPPORTS_WEXTRA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

check_cxx_compiler_flag("-Wdouble-promotion" COMPILER_SUPPORTS_WDOUBLE_PROMOTION)
if(COMPILER_SUPPORTS_WDOUBLE_PROMOTION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion")
endif()


check_cxx_compiler_flag("-Wfloat-conversion" COMPILER_SUPPORTS_WFLOAT_CONVERSION)
if(COMPILER_SUPPORTS_WFLOAT_CONVERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
endif()

# Define test targets if required
enable_testing()
if (SPHERICART_BUILD_TESTS)
    add_subdirectory(tests)
endif()


#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sphericart-config-version.in.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sphericart-config-version.cmake"
    @ONLY
)

install(TARGETS sphericart
    EXPORT sphericart-targets
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/sphericart-config.in.cmake"
    "${PROJECT_BINARY_DIR}/sphericart-config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/sphericart
)

install(EXPORT sphericart-targets DESTINATION ${LIB_INSTALL_DIR}/cmake/sphericart)
install(FILES "${PROJECT_BINARY_DIR}/sphericart-config-version.cmake"
              "${PROJECT_BINARY_DIR}/sphericart-config.cmake"
        DESTINATION ${LIB_INSTALL_DIR}/cmake/sphericart)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})
