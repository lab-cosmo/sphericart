cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    sphericart
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://example.com/"
    LANGUAGES C
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    sphericart_sphericart
    source/sphericart.c
)
add_library(sphericart::sphericart ALIAS sphericart_sphericart)

include(GenerateExportHeader)
generate_export_header(
    sphericart_sphericart
    BASE_NAME sphericart
    EXPORT_FILE_NAME export/sphericart/sphericart_export.h
)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(sphericart_sphericart PUBLIC SPHERICART_STATIC_DEFINE)
endif()

set_target_properties(
    sphericart_sphericart PROPERTIES
    C_VISIBILITY_PRESET hidden
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME sphericart
    OUTPUT_NAME sphericart
)

target_include_directories(
    sphericart_sphericart ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    sphericart_sphericart SYSTEM
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(sphericart_sphericart PUBLIC c_std_99)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${sphericart_DEVELOPER_MODE}")
  if(BUILD_EXAMPLES)
    add_subdirectory(example)
  endif()
endif()

# ---- Developer mode ----

if(NOT sphericart_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of sphericart"
  )
endif()

include(cmake/dev-mode.cmake)
