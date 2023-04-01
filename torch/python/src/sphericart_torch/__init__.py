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
torch.classes.load_library(_lib_path())


class SphericalHarmonics:
    def __init__(self, l_max, normalized=False):
        self._l_max = l_max
        self._sph = torch.classes.sphericart_torch.SphericalHarmonics(l_max, normalized)

    def compute(self, xyz):
        return self._sph.compute(xyz)

    def compute_with_gradients(self, xyz):
        return self._sph.compute_with_gradients(xyz)


__all__ = ["SphericalHarmonics"]
