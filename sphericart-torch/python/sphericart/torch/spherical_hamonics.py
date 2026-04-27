from typing import Tuple

import torch
from torch import Tensor


class SphericalHarmonics(torch.nn.Module):
    """
    Spherical harmonics calculator, which computes the real spherical harmonics
    :math:`Y^m_l` up to degree ``l_max``. The calculated spherical harmonics
    are consistent with the definition of real spherical harmonics from Wikipedia.

    This class can be used similarly to :py:class:`sphericart.SphericalHarmonics`
    (its Python/NumPy counterpart).

    >>> xyz = xyz.detach().clone().requires_grad_()
    >>> sh = sphericart.torch.SphericalHarmonics(l_max=8)
    >>> sh_values = sh(xyz)  # or sh.compute(xyz)
    >>> sh_values.sum().backward()
    >>> torch.allclose(xyz.grad, sh_grads.sum(axis=-1))
    True

    Reverse-mode differentiation with respect to ``xyz`` supports single and
    double backpropagation.

    Alternatively, the class allows to return explicit forward gradients and/or
    Hessians of the spherical harmonics. For example:

    >>> import torch
    >>> import sphericart.torch
    >>> sh = sphericart.torch.SphericalHarmonics(l_max=8)
    >>> xyz = torch.rand(size=(10, 3))
    >>> sh_values, sh_grads = sh.compute_with_gradients(xyz)
    >>> sh_grads.shape
    torch.Size([10, 3, 81])

    This class supports TorchScript and ``torch.compile``.

    :param l_max:
        the maximum degree of the spherical harmonics to be calculated
    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(self, l_max: int):
        super().__init__()
        self._l_max = l_max

    def forward(self, xyz: Tensor) -> Tensor:
        """
        Calculates the spherical harmonics for a set of 3D points.

        The coordinates should be stored in the ``xyz`` array.
        The type of the entries of ``xyz`` determines the precision used,
        and the device the tensor is stored on determines whether the
        CPU or CUDA implementation is used for the calculation backend.
        It supports single and double reverse-mode differentiation with
        respect to ``xyz``.

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
        return torch.ops.sphericart_torch.spherical_harmonics(xyz, self._l_max)

    def compute(self, xyz: Tensor) -> Tensor:
        """Equivalent to ``forward``"""
        return self.forward(xyz)

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
        return torch.ops.sphericart_torch.spherical_harmonics_with_gradients(
            xyz, self._l_max
        )

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
        return torch.ops.sphericart_torch.spherical_harmonics_with_hessians(
            xyz, self._l_max
        )

    def omp_num_threads(self):
        """Returns the number of threads available for calculations on the CPU."""
        return torch.ops.sphericart_torch.spherical_harmonics_omp_num_threads(
            self._l_max
        )

    def l_max(self):
        """Returns the maximum angular momentum setting for this calculator."""
        return self._l_max


class SolidHarmonics(torch.nn.Module):
    """
    Solid harmonics calculator, up to degree ``l_max``.

    This class computes the solid harmonics, a non-normalized form of the real
    spherical harmonics, i.e. :math:`r^lY^m_l`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points.

    The usage of this class is identical to :py:class:`sphericart.SphericalHarmonics`.

    :param l_max:
        the maximum degree of the spherical harmonics to be calculated
    :return: a calculator, in the form of a SolidHarmonics object
    """

    def __init__(self, l_max: int):
        super().__init__()
        self._l_max = l_max

    def forward(self, xyz: Tensor) -> Tensor:
        """See :py:meth:`SphericalHarmonics.forward`"""
        return torch.ops.sphericart_torch.solid_harmonics(xyz, self._l_max)

    def compute(self, xyz: Tensor) -> Tensor:
        """Equivalent to ``forward``"""
        return self.forward(xyz)

    def compute_with_gradients(self, xyz: Tensor) -> Tuple[Tensor, Tensor]:
        """See :py:meth:`SphericalHarmonics.compute_with_gradients`"""
        return torch.ops.sphericart_torch.solid_harmonics_with_gradients(
            xyz, self._l_max
        )

    def compute_with_hessians(self, xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """See :py:meth:`SphericalHarmonics.compute_with_hessians`"""
        return torch.ops.sphericart_torch.solid_harmonics_with_hessians(
            xyz, self._l_max
        )

    def omp_num_threads(self):
        """Returns the number of threads available for calculations on the CPU."""
        return torch.ops.sphericart_torch.solid_harmonics_omp_num_threads(self._l_max)

    def l_max(self):
        """Returns the maximum angular momentum setting for this calculator."""
        return self._l_max
