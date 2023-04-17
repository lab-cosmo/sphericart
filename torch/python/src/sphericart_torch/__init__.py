import math
import os
import sys
from typing import List, Optional, Tuple, Union


import torch


from ._build_torch_version import BUILD_TORCH_VERSION
import re


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
        f"Trying to load sphericart_torch with torch v{torch.__version__}, "
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


def e3nn_wrapper(
    l_list: Union[List[int], int],
    x: torch.Tensor,
    normalize: Optional[bool] = False,
    normalization: Optional[str] = "integral",
) -> torch.Tensor:
    """Provides an interface that is similar to `e3nn.o3.spherical_harmonics()`
    but uses SphericalHarmonics.compute(). Uses the same ordering of the
    [x,y,z] axes, and supports the same options for input and harmonics
    normalization. However,
    """

    if not hasattr(l_list, "__len__"):
        l_list = [l_list]
    l_max = max(l_list)
    sh = SphericalHarmonics(l_max, normalized=normalize).compute(
        x[:, [2, 0, 1]]
    )[0]

    if normalization != "integral":
        sh *= math.sqrt(4 * math.pi)

    sh_list = []
    for l in l_list:
        shl = sh[:, l * l : (l + 1) * (l + 1)]
        if normalization == "norm":
            shl *= math.sqrt(1 / (2 * l + 1))
        sh_list.append(shl)
    sh = torch.cat(sh_list, dim=-1)

    return sh


def patch_e3nn(e3nn_module):
    """Patches the e3nn module so that `sphericart_torch.e3nn_wrapper`
    is called in lieu of the built-in function."""

    e3nn_module.o3.spherical_harmonics = e3nn_wrapper


__all__ = ["SphericalHarmonics", "patch_e3nn"]
