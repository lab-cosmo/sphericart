import ctypes
import warnings
from pathlib import Path

from packaging import version

import jax

from .spherical_harmonics import solid_harmonics, spherical_harmonics  # noqa: F401


def get_minimum_cuda_version_for_jax(jax_version):
    """
    Get the minimum required CUDA version for a specific JAX version.

    Args:
        jax_version (str): Installed JAX version, e.g., '0.4.11'.

    Returns:
        tuple: Minimum required CUDA version as (major, minor), e.g., (11, 8).
    """
    # Define ranges of JAX versions and their corresponding minimum CUDA versions
    version_ranges = [
        (
            version.parse("0.5.0"),
            version.parse("999.999.999"),
            (12, 1),
        ),  # JAX 0.4.26 and later: CUDA 12.1+
    ]

    jax_ver = version.parse(jax_version)

    # Find the appropriate CUDA version range
    for start, end, cuda_version in version_ranges:
        if start <= jax_ver <= end:
            return cuda_version

    raise ValueError(f"Unsupported JAX version: {jax_version}")


def _load_shared_library(glob_pattern: str) -> ctypes.CDLL:
    lib_dir = Path(__file__).resolve().parent / "lib"
    matches = sorted(lib_dir.glob(glob_pattern))
    if not matches:
        raise ImportError(
            f"Could not find shared library matching {glob_pattern} in {lib_dir}"
        )
    return ctypes.cdll.LoadLibrary(str(matches[0]))


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

_cpu_lib = _load_shared_library("libsphericart_jax_cpu*")
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
    _cuda_lib = _load_shared_library("libsphericart_jax_cuda*")
    has_sphericart_jax_cuda = True
    _register_targets(_cuda_lib, _CUDA_TARGETS, platform="CUDA")
except Exception:
    has_sphericart_jax_cuda = False
    _cuda_lib = None


def get_cuda_runtime_version():
    """Return the host CUDA runtime version as {"major": int, "minor": int}."""
    if not has_sphericart_jax_cuda or _cuda_lib is None:
        raise ImportError(
            "Trying to use sphericart-jax on CUDA, but sphericart-jax was installed "
            "without CUDA support. Please re-install sphericart-jax with CUDA support."
        )
    major = ctypes.c_int()
    minor = ctypes.c_int()
    fn = _cuda_lib.sphericart_jax_get_cuda_runtime_version
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    fn.restype = None
    fn(ctypes.byref(major), ctypes.byref(minor))
    return {"major": int(major.value), "minor": int(minor.value)}


if has_sphericart_jax_cuda:
    cuda_version = get_cuda_runtime_version()
    cuda_version = (cuda_version["major"], cuda_version["minor"])
    jax_version = jax.__version__
    required_version = get_minimum_cuda_version_for_jax(jax_version)
    if cuda_version < required_version:
        warnings.warn(
            "The installed CUDA Toolkit version is "
            f"{cuda_version[0]}.{cuda_version[1]}, which "
            f"is not compatible with the installed JAX version {jax_version}. "
            "The minimum required CUDA Toolkit for your JAX version "
            f"is {required_version[0]}.{required_version[1]}. "
            "You might have to upgrade your CUDA Toolkit to meet the requirements, "
            "or downgrade JAX to a compatible version.",
            stacklevel=2,
        )
