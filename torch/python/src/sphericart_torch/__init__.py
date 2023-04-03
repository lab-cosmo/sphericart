import os
import sys
from typing import Optional, Tuple

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
    """
    Spherical harmonics calculator, up to degree ``l_max``.

    By default, this class computes a non-normalized form of the real spherical
    harmonics, i.e. :math:`r^l Y^l_m(r)`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points.
    ``normalize=True`` can be set to compute :math:`Y^l_m(r)`.

    :param l_max: the maximum degree of the spherical harmonics to be calculated
    :param normalized: whether to normalize the spherical harmonics (default: False)

    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(self, l_max: int, normalized: bool = False):
        self._l_max = l_max
        self._sph = torch.classes.sphericart_torch.SphericalHarmonics(l_max, normalized)

    def compute(
        self, xyz: torch.Tensor, gradients: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculates the spherical harmonics for a set of 3D points, whose
        coordinates are in the ``xyz`` array. If ``xyz`` has `requires_grad = True`
        it stores the forward derivatives which are then used in the backward
        pass.

        :param xyz:
            The Cartesian coordinates of the 3D points, as a `torch.Tensor` with
            shape ``(n_samples, 3)``.

        :return:
            A tuple containing two values:

            * A tensor of shape ``(n_samples, (l_max+1)**2)`` containing all the
              spherical harmonics up to degree `l_max` in lexicographic order.
              For example, if ``l_max = 2``, The last axis will correspond to
              spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
              1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * Either ``None`` if ``gradients=False`` or, if ``gradients=True``,
              a tensor of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
              the spherical harmonics' derivatives up to degree ``l_max``. The
              last axis is organized in the same way as in the spherical
              harmonics return array, while the second-to-last axis refers to
              derivatives in the the x, y, and z directions, respectively.
        """

        if gradients:
            return self._sph.compute_with_gradients(xyz)
        else:
            return self._sph.compute(xyz), None


__all__ = ["SphericalHarmonics"]
