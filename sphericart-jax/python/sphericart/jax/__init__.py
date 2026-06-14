import ctypes
import glob
import os
import re
import sys
from collections import namedtuple

import jax

from .spherical_harmonics import solid_harmonics, spherical_harmonics  # noqa: F401


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version_string):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version_string)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError(f"Invalid version string format: {version_string}")


_HERE = os.path.realpath(os.path.dirname(__file__))


def _get_lib_dir():
    jax_version = parse_version(jax.__version__)
    expected_prefix = os.path.join(
        _HERE, f"jax-{jax_version.major}.{jax_version.minor}.{jax_version.patch}"
    )
    if os.path.exists(expected_prefix):
        return expected_prefix

    # gather which jax version(s) the current install was built
    # with to create the error message
    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "jax-*")):
        existing_versions.append(os.path.basename(prefix)[4:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load sphericart-jax with jax v{jax.__version__}, "
            f"but it was compiled against jax v{existing_versions[0]}, which "
            "is not ABI compatible"
        )
    else:
        all_versions = ", ".join(map(lambda v: f"v{v}", existing_versions))
        raise ImportError(
            f"Trying to load sphericart-jax with jax v{jax.__version__}, "
            f"we found builds for jax {all_versions}; which are not ABI compatible.\n"
            "You can try to re-install from source with "
            "`pip install sphericart-jax --no-binary=sphericart-jax`"
        )


_LIB_DIR = _get_lib_dir()


def _lib_path(name: str) -> str:
    if sys.platform.startswith("darwin"):
        path = os.path.join(_LIB_DIR, "lib", f"lib{name}.dylib")
    elif sys.platform.startswith("linux"):
        path = os.path.join(_LIB_DIR, "lib", f"lib{name}.so")
    elif sys.platform.startswith("win"):
        path = os.path.join(_LIB_DIR, "bin", f"{name}.dll")
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if os.path.isfile(path):
        return path
    else:
        raise ImportError(f"Could not find shared library at {path}")


def _register_targets(lib: ctypes.CDLL, target_names: list[str], *, platform: str):
    for name in target_names:
        fn = getattr(lib, name)
        jax.ffi.register_ffi_target(
            name,
            jax.ffi.pycapsule(fn),
            platform=platform,
            api_version=1,  # typed FFI
        )


# === CPU targets ===

_CPU_TARGETS = [
    "cpu_spherical_f32",
    "cpu_spherical_f64",
    "cpu_dspherical_f32",
    "cpu_dspherical_f64",
    "cpu_ddspherical_f32",
    "cpu_ddspherical_f64",
    "cpu_solid_f32",
    "cpu_solid_f64",
    "cpu_dsolid_f32",
    "cpu_dsolid_f64",
    "cpu_ddsolid_f32",
    "cpu_ddsolid_f64",
]

_cpu_lib = ctypes.cdll.LoadLibrary(_lib_path("sphericart_jax_cpu"))
_register_targets(_cpu_lib, _CPU_TARGETS, platform="cpu")


# === CUDA targets ===

has_sphericart_jax_cuda = False
_cuda_lib = None

_CUDA_TARGETS = [
    "cuda_spherical_f32",
    "cuda_spherical_f64",
    "cuda_dspherical_f32",
    "cuda_dspherical_f64",
    "cuda_ddspherical_f32",
    "cuda_ddspherical_f64",
    "cuda_solid_f32",
    "cuda_solid_f64",
    "cuda_dsolid_f32",
    "cuda_dsolid_f64",
    "cuda_ddsolid_f32",
    "cuda_ddsolid_f64",
]

try:
    _cuda_lib = ctypes.cdll.LoadLibrary(_lib_path("sphericart_jax_cuda"))
    has_sphericart_jax_cuda = True
    _register_targets(_cuda_lib, _CUDA_TARGETS, platform="CUDA")
except Exception:
    has_sphericart_jax_cuda = False
    _cuda_lib = None
