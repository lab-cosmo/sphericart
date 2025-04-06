from typing import Tuple

import torch
from torch import Tensor


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
