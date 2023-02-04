cmake_minimum_required(VERSION 3.14)

project(sphericartTests LANGUAGES C)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)
include(../cmake/windows-set-path.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(sphericart REQUIRED)
  enable_testing()
endif()

# ---- Tests ----

add_executable(sphericart_test source/sphericart_test.c)
target_link_libraries(sphericart_test PRIVATE sphericart::sphericart)
target_compile_features(sphericart_test PRIVATE c_std_99)

add_test(NAME sphericart_test COMMAND sphericart_test)
windows_set_path(sphericart_test sphericart::sphericart)

# ---- End-of-file commands ----

add_folders(Test)
