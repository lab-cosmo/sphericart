# this is a custom Python build backend wrapping setuptool's to set a
# specific torch version as a build dependency, based on an environment
# variable
import os

from setuptools import build_meta

TORCH_VERSION = os.environ.get("SPHERICART_TORCH_TORCH_VERSION")

if TORCH_VERSION is not None:
    TORCH_DEP = f"torch =={TORCH_VERSION}"
else:
    TORCH_DEP = "torch >=1.13"


prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [TORCH_DEP]


get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
