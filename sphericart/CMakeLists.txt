cmake_minimum_required(VERSION 3.10)

project(sphericart CXX)

file(READ ${PROJECT_SOURCE_DIR}/VERSION SPHERICART_VERSION)
string(STRIP ${SPHERICART_VERSION} SPHERICART_VERSION)
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" SPHERICART_VERSION_MAJOR "${SPHERICART_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" SPHERICART_VERSION_MINOR "${SPHERICART_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" SPHERICART_VERSION_PATCH "${SPHERICART_VERSION}")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

OPTION(SPHERICART_BUILD_TESTS "Build and run tests for Sphericart" ON) # TODO: change default to OFF
OPTION(SPHERICART_OPENMP "Try to use OpenMP when compiling Sphericart" ON)
OPTION(SPHERICART_BUILD_TORCH "Build the torch bindings" OFF)

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

# we need to compile sphericart with C++17 for if constexpr
target_compile_features(sphericart PUBLIC cxx_std_17)

target_include_directories(sphericart PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create a header defining SPHERICAR_EXPORT for exported classes/functions
set_target_properties(sphericart PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
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

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    message(STATUS "march=native is enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
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

# Define test targets if required
enable_testing()
if (SPHERICART_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (SPHERICART_BUILD_TORCH)
    add_subdirectory(../torch torch)
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
