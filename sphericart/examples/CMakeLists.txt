add_executable(example_cpp example.cpp)
add_executable(example_c example.c)
target_link_libraries(example_cpp sphericart)
target_link_libraries(example_c sphericart)

set_target_properties(example_c PROPERTIES
  LINKER_LANGUAGE C
)

