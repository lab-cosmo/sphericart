if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/sphericart-${PROJECT_VERSION}"
      CACHE PATH ""
  )
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package sphericart)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT sphericart_Development
)

install(
    TARGETS sphericart_sphericart
    EXPORT sphericartTargets
    RUNTIME #
    COMPONENT sphericart_Runtime
    LIBRARY #
    COMPONENT sphericart_Runtime
    NAMELINK_COMPONENT sphericart_Development
    ARCHIVE #
    COMPONENT sphericart_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    sphericart_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(sphericart_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${sphericart_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT sphericart_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${sphericart_INSTALL_CMAKEDIR}"
    COMPONENT sphericart_Development
)

install(
    EXPORT sphericartTargets
    NAMESPACE sphericart::
    DESTINATION "${sphericart_INSTALL_CMAKEDIR}"
    COMPONENT sphericart_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
