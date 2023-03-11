import os
import sys

import torch

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
torch.ops.load_library(_lib_path())


def spherical_harmonics(
    l_max: int,
    xyz: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute spherical harmonics, taking the input directions as a torch Tensor

    By default, this computes un-normalized, cartesian spherical harmonics.

    :param l_max: max L to use when computing spherical harmonics
    :param xyz: torch tensor containing the directions
    :param normalize: should we compute normalized or non-normalized cartesian
        spherical harmonics
    :return: Spherical harmonics corresponding to ``xyz``, up to order ``l_max``
    """
    return torch.ops.sphericart.spherical_harmonics(
        l_max=l_max,
        xyz=xyz,
        normalize=normalize,
    )


__all__ = ["spherical_harmonics"]
