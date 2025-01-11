import math
from types import ModuleType
from typing import List, Optional, Union

import torch
from torch import Tensor

from .spherical_hamonics import SolidHarmonics, SphericalHarmonics


def e3nn_spherical_harmonics(
    l_list: Union[List[int], int],
    x: Tensor,
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
        :math:`Y^m_l` should be computed. All values up to the maximum
        l value are computed, so this may be inefficient for use cases
        requiring a single, or few, angular momentum channels.
    :param x:
        A ``torch.Tensor`` containing the coordinates, in the same format
        expected by the ``e3nn`` function.
    :param normalize:
        Flag specifying whether the input positions should be normalized
        (resulting in the computation of the spherical harmonics :math:`Y^m_l`),
        or whether the function should compute the solid harmonics
        :math:`r^lY^m_l`.
    :param normalization:
        String that can be "integral", "norm", "component", that controls
        a further scaling of the :math:`Y^m_l`. See the
        documentation of :py:func:`e3nn.o3.spherical_harmonics()`
        for a detailed explanation of the different conventions.
    """

    if not hasattr(l_list, "__len__"):
        l_list = [l_list]
    l_max = max(l_list)
    is_range_lmax = list(l_list) == list(range(l_max + 1))

    if normalize:
        sh = SphericalHarmonics(l_max)(
            torch.index_select(
                x, 1, torch.tensor([2, 0, 1], dtype=torch.long, device=x.device)
            )
        )
    else:
        sh = SolidHarmonics(l_max)(
            torch.index_select(
                x, 1, torch.tensor([2, 0, 1], dtype=torch.long, device=x.device)
            )
        )
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
    "SolidHarmonics",
    "e3nn_spherical_harmonics",
    "patch_e3nn",
    "unpatch_e3nn",
]
