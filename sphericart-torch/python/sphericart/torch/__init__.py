import os
import sys

import torch

from ._build_torch_version import BUILD_TORCH_VERSION
import re

from .spherical_hamonics import SphericalHarmonics, SolidHarmonics  # noqa: F401
from .e3nn import patch_e3nn, unpatch_e3nn, e3nn_spherical_harmonics  # noqa: F401


def parse_version_string(version_string):
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_string)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def torch_version_compatible(actual, required):
    actual_version_tuple = parse_version_string(actual)
    required_version_tuple = parse_version_string(required)

    if actual_version_tuple[0] != required_version_tuple[0]:
        return False
    elif actual_version_tuple[1] != required_version_tuple[1]:
        return False
    else:
        return True


if not torch_version_compatible(torch.__version__, BUILD_TORCH_VERSION):
    raise ImportError(
        f"Trying to load sphericart-torch with torch v{torch.__version__}, "
        f"but it was compiled against torch v{BUILD_TORCH_VERSION}, which "
        "is not ABI compatible"
    )


_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    if sys.platform.startswith("darwin"):
        name = "libsphericart_torch.dylib"
    elif sys.platform.startswith("linux"):
        name = "libsphericart_torch.so"
    elif sys.platform.startswith("win"):
        name = "sphericart_torch.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")

    path = os.path.join(os.path.join(_HERE, "lib"), name)

    if os.path.isfile(path):
        return path

    raise ImportError("Could not find sphericart_torch shared library at " + path)


# load the C++ operators and custom classes
torch.classes.load_library(_lib_path())
