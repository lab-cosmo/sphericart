import math
import os
import sys
from types import ModuleType
from collections import namedtuple
from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor

import re
import glob


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    torch_version = parse_version(torch.__version__)
    expected_prefix = os.path.join(
        _HERE, f"../torch-{torch_version.major}.{torch_version.minor}"
    )
    if os.path.exists(expected_prefix):
        if sys.platform.startswith("darwin"):
            path = os.path.join(expected_prefix, "lib", "libsphericart_torch.dylib")
        elif sys.platform.startswith("linux"):
            path = os.path.join(expected_prefix, "lib", "libsphericart_torch.so")
        elif sys.platform.startswith("win"):
            path = os.path.join(expected_prefix, "bin", "sphericart_torch.dll")
        else:
            raise ImportError("Unknown platform. Please edit this file")

        if os.path.isfile(path):
            # if windows:
            #     _check_dll(path)
            return path
        else:
            raise ImportError(
                "Could not find sphericart_torch shared library at " + path
            )

    # gather which torch version(s) the current install was built
    # with to create the error message
    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "../torch-*")):
        existing_versions.append(os.path.basename(prefix)[11:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load sphericart-torch with torch v{torch.__version__}, "
            f"but it was compiled against torch v{existing_versions[0]}, which "
            "is not ABI compatible"
        )
    else:
        all_versions = ", ".join(map(lambda version: f"v{version}", existing_versions))
        raise ImportError(
            f"Trying to load sphericart-torch with torch v{torch.__version__}, "
            f"we found builds for torch {all_versions}; which are not ABI compatible.\n"
            "You can try to re-install from source with "
            "`pip install sphericart-torch --no-binary=sphericart-torch`"
        )


# load the C++ operators and custom classes
torch.classes.load_library(_lib_path())


# This is a workaround to provide docstrings for the SphericalHarmonics class,
# even though it is defined as a C++ TorchScript object (and we can not figure
# out a way to extract docstrings for either classes or methods from the C++
# code). The class reproduces the API of the TorchScript class, but has empty
# functions. Instead, when __new__ is called, an instance of the TorchScript
# class is directly returned.
class SphericalHarmonics(torch.nn.Module):
    """
    Spherical harmonics calculator, which computes the real spherical harmonics
    :math:`Y^m_l` up to degree ``l_max``. The calculated spherical harmonics
    are consistent with the definition of real spherical harmonics from Wikipedia.

    This class can be used similarly to :py:class:`sphericart.SphericalHarmonics`
    (its Python/NumPy counterpart). If the class is called directly, the outputs
    support single and double backpropagation.

    >>> xyz = xyz.detach().clone().requires_grad_()
    >>> sh = sphericart.torch.SphericalHarmonics(l_max=8)
    >>> sh_values = sh(xyz)  # or sh.compute(xyz)
    >>> sh_values.sum().backward()
    >>> torch.allclose(xyz.grad, sh_grads.sum(axis=-1))
    True

    By default, only single backpropagation with respect to ``xyz`` is
    enabled (this includes mixed second derivatives where ``xyz`` appears
    as only one of the differentiation steps). To activate support
    for double backpropagation with respect to ``xyz``, please set
    ``backward_second_derivatives=True`` at class creation. Warning: if
    ``backward_second_derivatives`` is not set to ``True`` and double
    differentiation with respect to ``xyz`` is requested, the results may
    be incorrect, but a warning will be displayed. This is necessary to
    provide optimal performance for both use cases. In particular, the
    following will happen:

    -   when using ``torch.autograd.grad`` as the second backpropagation
        step, a warning will be displayed and torch will raise an error.
    -   when using ``torch.autograd.grad`` with ``allow_unused=True`` as
        the second backpropagation step, the results will be incorrect
        and only a warning will be displayed.
    -   when using ``backward`` as the second backpropagation step, the
        results will be incorrect and only a warning will be displayed.
    -   when using ``torch.autograd.functional.hessian``, the results will
        be incorrect and only a warning will be displayed.

    Alternatively, the class allows to return explicit forward gradients and/or
    Hessians of the spherical harmonics. For example:

    >>> import torch
    >>> import sphericart.torch
    >>> sh = sphericart.torch.SphericalHarmonics(l_max=8)
    >>> xyz = torch.rand(size=(10,3))
    >>> sh_values, sh_grads = sh.compute_with_gradients(xyz)
    >>> sh_grads.shape
    torch.Size([10, 3, 81])

    This class supports TorchScript.

    :param l_max:
        the maximum degree of the spherical harmonics to be calculated
    :param backward_second_derivatives:
        if this parameter is set to ``True``, second derivatives of the spherical
        harmonics are calculated and stored during forward calls to ``compute``
        (provided that ``xyz.requires_grad`` is ``True``), making it possible to perform
        double reverse-mode differentiation with respect to ``xyz``. If ``False``, only
        the first derivatives will be computed and only a single reverse-mode
        differentiation step will be possible with respect to ``xyz``.

    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(
        self,
        l_max: int,
        backward_second_derivatives: bool = False,
    ):
        super().__init__()
        self.calculator = torch.classes.sphericart_torch.SphericalHarmonics(
            l_max, backward_second_derivatives
        )

    def forward(self, xyz: Tensor) -> Tensor:
        """
        Calculates the spherical harmonics for a set of 3D points.

        The coordinates should be stored in the ``xyz`` array. If ``xyz``
        has ``requires_grad = True`` it stores the forward derivatives which
        are then used in the backward pass.
        The type of the entries of ``xyz`` determines the precision used,
        and the device the tensor is stored on determines whether the
        CPU or CUDA implementation is used for the calculation backend.
        It always supports single reverse-mode differentiation, as well as
        double reverse-mode differentiation if ``backward_second_derivatives``
        was set to ``True`` during class creation.

        :param xyz:
            The Cartesian coordinates of the 3D points, as a `torch.Tensor` with
            shape ``(n_samples, 3)``.

        :return:
            A tensor of shape ``(n_samples, (l_max+1)**2)`` containing all the
            spherical harmonics up to degree `l_max` in lexicographic order.
            For example, if ``l_max = 2``, The last axis will correspond to
            spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
            1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
        """
        return self.calculator.compute(xyz)

    def compute(self, xyz: Tensor) -> Tensor:
        """Equivalent to ``forward``"""
        return self.calculator.compute(xyz)

    def compute_with_gradients(self, xyz: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculates the spherical harmonics for a set of 3D points,
        and also returns the forward-mode derivatives.

        The coordinates should be stored in the ``xyz`` array.
        The type of the entries of ``xyz`` determines the precision used,
        and the device the tensor is stored on determines whether the
        CPU or CUDA implementation is used for the calculation backend.
        Reverse-mode differentiation is not supported for this function.

        :param xyz:
            The Cartesian coordinates of the 3D points, as a `torch.Tensor` with
            shape ``(n_samples, 3)``.

        :return:
            A tuple that contains:

            * A ``(n_samples, (l_max+1)**2)`` tensor containing all the
              spherical harmonics up to degree ``l_max`` in lexicographic order.
              For example, if ``l_max = 2``, The last axis will correspond to
              spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
              1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * A tensor of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
              the spherical harmonics' derivatives up to degree ``l_max``. The
              last axis is organized in the same way as in the spherical
              harmonics return array, while the second-to-last axis refers to
              derivatives in the the x, y, and z directions, respectively.

        """
        return self.calculator.compute_with_gradients(xyz)

    def compute_with_hessians(self, xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the spherical harmonics for a set of 3D points,
        and also returns the forward derivatives and second derivatives.

        The coordinates should be stored in the ``xyz`` array.
        The type of the entries of ``xyz`` determines the precision used,
        and the device the tensor is stored on determines whether the
        CPU or CUDA implementation is used for the calculation backend.
        Reverse-mode differentiation is not supported for this function.

        :param xyz:
            The Cartesian coordinates of the 3D points, as a ``torch.Tensor`` with
            shape ``(n_samples, 3)``.

        :return:
            A tuple that contains:

            * A ``(n_samples, (l_max+1)**2)`` tensor containing all the
              spherical harmonics up to degree ``l_max`` in lexicographic order.
              For example, if ``l_max = 2``, The last axis will correspond to
              spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
              1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * A tensor of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
              the spherical harmonics' derivatives up to degree ``l_max``. The
              last axis is organized in the same way as in the spherical
              harmonics return array, while the second-to-last axis refers to
              derivatives in the the x, y, and z directions, respectively.
            * A tensor of shape ``(n_samples, 3, 3, (l_max+1)**2)`` containing all
              the spherical harmonics' second derivatives up to degree ``l_max``. The
              last axis is organized in the same way as in the spherical
              harmonics return array, while the two intermediate axes represent the
              hessian dimensions.

        """
        return self.calculator.compute_with_hessians(xyz)

    def omp_num_threads(self):
        """Returns the number of threads available for calculations on the CPU."""
        return self.calculator.omp_num_threads()

    def l_max(self):
        """Returns the maximum angular momentum setting for this calculator."""
        return self.calculator.l_max()


class SolidHarmonics(torch.nn.Module):
    """
    Solid harmonics calculator, up to degree ``l_max``.

    This class computes the solid harmonics, a non-normalized form of the real
    spherical harmonics, i.e. :math:`r^lY^m_l`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points.

    The usage of this class is identical to :py:class:`sphericart.SphericalHarmonics`.

    :param l_max:
        the maximum degree of the spherical harmonics to be calculated
    :param backward_second_derivatives:
        if this parameter is set to ``True``, second derivatives of the spherical
        harmonics are calculated and stored during forward calls to ``compute``
        (provided that ``xyz.requires_grad`` is ``True``), making it possible to perform
        double reverse-mode differentiation with respect to ``xyz``. If ``False``, only
        the first derivatives will be computed and only a single reverse-mode
        differentiation step will be possible with respect to ``xyz``.

    :return: a calculator, in the form of a SolidHarmonics object
    """

    def __init__(
        self,
        l_max: int,
        backward_second_derivatives: bool = False,
    ):
        super().__init__()
        self.calculator = torch.classes.sphericart_torch.SolidHarmonics(
            l_max, backward_second_derivatives
        )

    def forward(self, xyz: Tensor) -> Tensor:
        """See :py:meth:`SphericalHarmonics.forward`"""
        return self.calculator.compute(xyz)

    def compute(self, xyz: Tensor) -> Tensor:
        """Equivalent to ``forward``"""
        return self.calculator.compute(xyz)

    def compute_with_gradients(self, xyz: Tensor) -> Tuple[Tensor, Tensor]:
        """See :py:meth:`SphericalHarmonics.compute_with_gradients`"""
        return self.calculator.compute_with_gradients(xyz)

    def compute_with_hessians(self, xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """See :py:meth:`SphericalHarmonics.compute_with_hessians`"""
        return self.calculator.compute_with_hessians(xyz)

    def omp_num_threads(self):
        """Returns the number of threads available for calculations on the CPU."""
        return self.calculator.omp_num_threads()

    def l_max(self):
        """Returns the maximum angular momentum setting for this calculator."""
        return self.calculator.l_max()


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
