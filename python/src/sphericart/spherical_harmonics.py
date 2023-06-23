import ctypes
from typing import Tuple

import numpy as np

from ._c_lib import _get_library


class SphericalHarmonics:
    """
    Spherical harmonics calculator, up to degree ``l_max``.

    By default, this class computes a non-normalized form of the real spherical
    harmonics, i.e. :math:`r^l Y^l_m`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points.
    ``normalize=True`` can be set to compute :math:`Y^l_m`.

    In order to minimize the cost of each call, the `SphericalHarmonics` object
    computes prefactors and initializes buffers upon creation

    >>> import numpy as np
    >>> import sphericart as sc
    >>> sh = sc.SphericalHarmonics(l_max=8, normalized=False)

    Then, the :py:func:`compute` method can be called on an array of 3D
    Cartesian points to compute the spherical harmonics

    >>> xyz = np.random.normal(size=(10,3))
    >>> sh_values = sh.compute(xyz)
    >>> sh_values.shape
    (10, 81)

    In order to also compute derivatives, you can use

    >>> sh_values, sh_grads = sh.compute_with_gradients(xyz)
    >>> sh_grads.shape
    (10, 3, 81)

    which returns the gradient as a tensor with size
    `(n_samples, 3, (l_max+1)**2)`.

    :param l_max: the maximum degree of the spherical harmonics to be calculated
    :param normalized: whether to normalize the spherical harmonics (default: False)

    :return: a calculator, in the form of a `SphericalHarmonics` object
    """

    def __init__(self, l_max: int, normalized: bool = False):
        self._l_max = l_max

        self._lib = _get_library()

        # intialize both a double precision and a single-precision calculator.
        # we will decide which one to use depending on the dtype of the data
        self._calculator = self._lib.sphericart_new(l_max, normalized)
        self._calculator_f = self._lib.sphericart_new_f(l_max, normalized)

        # this allows to check the number of threads that are used
        # it is 1 if there is no OpenMP available
        self._omp_num_threads = self._lib.sphericart_omp_num_threads(self._calculator)

    def __del__(self):
        self._lib.sphericart_delete(self._calculator)
        self._lib.sphericart_delete_f(self._calculator_f)

    def compute(self, xyz: np.ndarray) -> np.ndarray:
        """
        Calculates the spherical harmonics for a set of 3D points, whose
        coordinates are in the ``xyz`` array.

        >>> import numpy as np
        >>> import sphericart as sc
        >>> sh = sc.SphericalHarmonics(l_max=8, normalized=False)
        >>> xyz = np.random.normal(size=(10,3))
        >>> sh_values = sh.compute(xyz)
        >>> sh_values.shape
        (10, 81)

        :param xyz:
            The Cartesian coordinates of the 3D points, as an array with
            shape ``(n_samples, 3)``

        :return:
            An array of shape ``(n_samples, (l_max+1)**2)`` containing all the
            spherical harmonics up to degree `l_max` in lexicographic order.
            For example, if ``l_max = 2``, The last axis will correspond to
            spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
            1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
        """

        if not isinstance(xyz, np.ndarray):
            raise TypeError("xyz must be a numpy array")

        if not (xyz.dtype == np.float32 or xyz.dtype == np.float64):
            raise TypeError("xyz must be a numpy array of 32 or 64-bit floats")

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz array must be a `N x 3` array")

        # make xyz contiguous before taking a pointer to it
        xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        xyz_length = n_samples * 3

        sph = np.empty((n_samples, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        sph_length = n_samples * (self._l_max + 1) ** 2

        if xyz.dtype == np.float64:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            self._lib.sphericart_compute_array(
                self._calculator, xyz_ptr, xyz_length, sph_ptr, sph_length
            )
        elif xyz.dtype == np.float32:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self._lib.sphericart_compute_array_f(
                self._calculator_f, xyz_ptr, xyz_length, sph_ptr, sph_length
            )

        return sph

    def compute_with_gradients(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the spherical harmonics for a set of 3D points, whose
        coordinates are in the ``xyz`` array, together with their Cartesian
        derivatives.

        >>> import numpy as np
        >>> import sphericart as sc
        >>> sh = sc.SphericalHarmonics(l_max=8, normalized=False)
        >>> xyz = np.random.normal(size=(10,3))
        >>> sh_values, sh_grads = sh.compute_with_gradients(xyz)
        >>> sh_grads.shape
        (10, 3, 81)

        :param xyz:
            The Cartesian coordinates of the 3D points, as an array with
            shape ``(n_samples, 3)``.

        :return:
            A tuple containing:
            * an array of shape ``(n_samples, (l_max+1)**2)`` containing all the
            spherical harmonics up to degree `l_max` in lexicographic order.
            For example, if ``l_max = 2``, The last axis will correspond to
            spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
            1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * An array of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
            the spherical harmonics' derivatives up to degree ``l_max``. The
            last axis is organized in the same way as in the spherical
            harmonics return array, while the second-to-last axis refers to
            derivatives in the the x, y, and z directions, respectively.

        """

        if not isinstance(xyz, np.ndarray):
            raise TypeError("xyz must be a numpy array")

        if not (xyz.dtype == np.float32 or xyz.dtype == np.float64):
            raise TypeError("xyz must be a numpy array of 32 or 64-bit floats")

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz array must be a `N x 3` array")

        # make xyz contiguous before taking a pointer to it
        xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        xyz_length = n_samples * 3

        sph = np.empty((n_samples, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        sph_length = n_samples * (self._l_max + 1) ** 2
        dsph = np.empty((n_samples, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        dsph_length = n_samples * 3 * (self._l_max + 1) ** 2

        if xyz.dtype == np.float64:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            self._lib.sphericart_compute_array_with_gradients(
                self._calculator,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                dsph_ptr,
                dsph_length,
            )
        elif xyz.dtype == np.float32:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self._lib.sphericart_compute_array_with_gradients_f(
                self._calculator_f,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                dsph_ptr,
                dsph_length,
            )

        return sph, dsph

    def compute_with_hessians(
        self, xyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the spherical harmonics for a set of 3D points, whose
        coordinates are in the ``xyz`` array, together with their Cartesian
        derivatives and second derivatives.

        >>> import numpy as np
        >>> import sphericart as sc
        >>> sh = sc.SphericalHarmonics(l_max=8, normalized=False)
        >>> xyz = np.random.normal(size=(10,3))
        >>> sh_values, sh_grads, sh_hessians = sh.compute_with_hessians(xyz)
        >>> sh_hessians.shape
        (10, 3, 3, 81)

        :param xyz:
            The Cartesian coordinates of the 3D points, as an array with
            shape ``(n_samples, 3)``.

        :return:
            A tuple containing:
            * an array of shape ``(n_samples, (l_max+1)**2)`` containing all the
            spherical harmonics up to degree `l_max` in lexicographic order.
            For example, if ``l_max = 2``, The last axis will correspond to
            spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
            1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * An array of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
            the spherical harmonics' derivatives up to degree ``l_max``. The
            last axis is organized in the same way as in the spherical
            harmonics return array, while the second-to-last axis refers to
            derivatives in the the x, y, and z directions, respectively.
            * An array of shape ``(n_samples, 3, 3, (l_max+1)**2)`` containing all
            the spherical harmonics' second derivatives up to degree ``l_max``. 
            The last axis is organized in the same way as in the spherical
            harmonics return array, while the two intermediate axes represent the
            hessian dimensions.

        """

        if not isinstance(xyz, np.ndarray):
            raise TypeError("xyz must be a numpy array")

        if not (xyz.dtype == np.float32 or xyz.dtype == np.float64):
            raise TypeError("xyz must be a numpy array of 32 or 64-bit floats")

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz array must be a `N x 3` array")

        # make xyz contiguous before taking a pointer to it
        xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        xyz_length = n_samples * 3

        sph = np.empty((n_samples, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        sph_length = n_samples * (self._l_max + 1) ** 2
        dsph = np.empty((n_samples, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        dsph_length = n_samples * 3 * (self._l_max + 1) ** 2
        ddsph = np.empty((n_samples, 3, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)
        ddsph_length = n_samples * 9 * (self._l_max + 1) ** 2

        if xyz.dtype == np.float64:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ddsph_ptr = ddsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            self._lib.sphericart_compute_array_with_hessians(
                self._calculator,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                dsph_ptr,
                dsph_length,
                ddsph_ptr,
                ddsph_length,
            )
        elif xyz.dtype == np.float32:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ddsph_ptr = ddsph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            self._lib.sphericart_compute_array_with_gradients_f(
                self._calculator_f,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                dsph_ptr,
                dsph_length,
                ddsph_ptr,
                ddsph_length,
            )

        return sph, dsph, ddsph
