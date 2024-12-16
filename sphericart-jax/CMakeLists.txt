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

find_package(pybind11 CONFIG REQUIRED)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(sphericart EXCLUDE_FROM_ALL)

# CPU op library
set(CPU_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/sphericart_jax_cpu.cpp)
pybind11_add_module(sphericart_jax_cpu ${CPU_SOURCES})
install(TARGETS sphericart_jax_cpu DESTINATION sphericart_jax_cpu)

target_link_libraries(sphericart_jax_cpu PUBLIC sphericart)
target_compile_features(sphericart_jax_cpu PUBLIC cxx_std_17)

target_include_directories(sphericart_jax_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS sphericart_jax_cpu
    LIBRARY DESTINATION "lib"
)

if(CMAKE_CUDA_COMPILER AND SPHERICART_ENABLE_CUDA)
    set(CUDA_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/sphericart_jax_cuda.cpp)
    pybind11_add_module(sphericart_jax_cuda ${CUDA_SOURCES})

    
    target_link_libraries(sphericart_jax_cuda PUBLIC sphericart)
    target_compile_features(sphericart_jax_cuda PUBLIC cxx_std_17)

    target_include_directories(sphericart_jax_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    install(TARGETS sphericart_jax_cuda DESTINATION sphericart_jax_cuda)
    install(TARGETS sphericart_jax_cuda
        LIBRARY DESTINATION "lib"
    )
endif()
