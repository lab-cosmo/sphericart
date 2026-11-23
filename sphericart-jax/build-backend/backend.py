# This is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies on jax/cmake when building the wheel and not the sdist
from setuptools import build_meta


# ==================================================================================== #
#                   Build backend functions definition                                 #
# ==================================================================================== #

# Use the default version of these
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


# Special dependencies to build the wheels
def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [
        "cmake >=3.30",
        # This should be the lowest supported version of `install_requires` in setup.py
        "jaxlib ==0.10.*",
    ]
