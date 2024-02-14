# this is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to metatensor-core, using the local version if it exists, and
# otherwise falling back to the one on PyPI.
import os

from setuptools import build_meta

TORCH_VERSION = os.environ.get("SPHERICART_TORCH_TORCH_VERSION")
CUDA_VERSION = os.environ.get("SPHERICART_TORCH_CUDA_VERSION")

if TORCH_VERSION is not None:
    # force a specific version of torch+cuda
    TORCH_DEP = f"torch =={TORCH_VERSION}"
    if CUDA_VERSION is not None:
        extra_index_url = f" --index-url https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"
        TORCH_DEP += extra_index_url
else:
    TORCH_DEP = "torch >=1.13"


prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [TORCH_DEP]


get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
