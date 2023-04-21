import math
import os
import sys
from types import ModuleType
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

class SphericalHarmonics(torch.nn.Module):
    """
    Spherical harmonics calculator, up to degree ``l_max``.

    By default, this class computes a non-normalized form of the real spherical
    harmonics, i.e. :math:`r^l Y^l_m(r)`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points.
    ``normalize=True`` can be set to compute :math:`Y^l_m(r)`.

    :param l_max:
        the maximum degree of the spherical harmonics to be calculated
    :param normalized:
        whether to normalize the spherical harmonics (default: False)

    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(self, l_max: int, normalized: bool = False):
        self._l_max = l_max
        self._sph = torch.classes.sphericart_torch.SphericalHarmonics(l_max, normalized)
        self._omp_num_threads = self._sph.get_omp_num_threads()
        super().__init__()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        sph =  self._sph.compute(xyz)
        return sph

    def compute(
        self, xyz: torch.Tensor, gradients: bool = False
    ) -> List[torch.Tensor]:
        """
        Calculates the spherical harmonics for a set of 3D points.

        The coordinates should be stored in the ``xyz`` array. If ``xyz``
        has `requires_grad = True` it stores the forward derivatives which
        are then used in the backward pass.
        The type of the entries of `xyz` determines the precision used,
        and the device the tensor is stored on determines whether the
        CPU or CUDA implementation is used for the calculation backend.

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
            return self._sph.compute(xyz), torch.empty_like(xyz)


def e3nn_spherical_harmonics(
    l_list: Union[List[int], int],
    x: torch.Tensor,
    normalize: Optional[bool] = False,
    normalization: Optional[str] = "integral",
) -> torch.Tensor:
    """
    Computes spherical harmonics with an interface similar to the e3nn package.

    Provides an interface that is similar to :py:func:`e3nn.o3.spherical_harmonics`
    but uses :py:class:`SphericalHarmonics` for the actual calculation.
    Uses the same ordering of the [x,y,z] axes, and supports the same options for
    input and harmonics normalization as :py:mod:`e3nn`. However, it does not support
    defining the irreps through a :py:class:`e3nn.o3._irreps.Irreps` or a string
    specification, but just as a single integer or a list of integers.

    :param l_list:
        Either a single integer or a list of integers specifying which
        :math:`Y^l_m` should be computed. All values up to the maximum
        l value are computed, so this may be inefficient for use cases
        requiring a single, or few, angular momentum channels.
    :param x:
        A `torch.Tensor` containing the coordinates, in the same format
        expected by the `e3nn` function.
    :param normalize:
        Flag specifying whether the input positions should be normalized,
        or whether the function should compute scaled :math:`\tilde{Y}^l_m`
    :param normalization:
        String that can be "integral", "norm", "component", that controls
        a further scaling of the :math:`Y_m^l`. See the
        documentation of :py:func:`e3nn.o3.spherical_harmonics()`
        for a detailed explanation of the different conventions.
    """

    if not hasattr(l_list, "__len__"):
        l_list = [l_list]
    l_max = max(l_list)
    is_range_lmax = list(l_list) == list(range(l_max + 1))

    sh = SphericalHarmonics(l_max, normalized=normalize).compute(x[:, [2, 0, 1]])[0]
    assert normalization in ["integral", "norm", "component"]
    if normalization != "integral":
        sh *= math.sqrt(4 * math.pi)

    if not is_range_lmax:
        sh_list = []
        for l in l_list:  # noqa E741
            shl = sh[:, l * l : (l + 1) * (l + 1)]
            if normalization == "norm":
                shl *= math.sqrt(1 / (2 * l + 1))
            sh_list.append(shl)
        sh = torch.cat(sh_list, dim=-1)
    elif normalization == "norm":
        for l in l_list:  # noqa E741
            sh[:, l * l : (l + 1) * (l + 1)] *= math.sqrt(1 / (2 * l + 1))

    return sh


_E3NN_SPH = None


def patch_e3nn(e3nn_module: ModuleType) -> None:
    """Patches the :py:mod:`e3nn` module so that
    :py:func:`sphericart_torch.e3nn_spherical_harmonics`
    is called in lieu of the built-in function.

    :param e3nn_module:
        The alias that has been chosen for the e3nn module,
        usually just ``e3nn``.
    """

    global _E3NN_SPH
    if _E3NN_SPH is not None:
        raise RuntimeError("It appears that e3nn has already been patched")

    _E3NN_SPH = e3nn_module.o3.spherical_harmonics
    e3nn_module.o3.spherical_harmonics = e3nn_spherical_harmonics


def unpatch_e3nn(e3nn_module: ModuleType) -> None:
    """Restore the original ``spherical_harmonics`` function
    in the :py:mod:`e3nn` module."""

    global _E3NN_SPH
    if _E3NN_SPH is None:
        raise RuntimeError("It appears that e3nn has not been patched")

    e3nn_module.o3.spherical_harmonics = _E3NN_SPH
    _E3NN_SPH = None


__all__ = [
    "SphericalHarmonics",
    "e3nn_spherical_harmonics",
    "patch_e3nn",
    "unpatch_e3nn",
]
