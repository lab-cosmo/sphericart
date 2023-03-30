import ctypes
from typing import Optional

import numpy as np

from ._c_lib import _get_library


class SphericalHarmonics:
    """
    Spherical harmonics calculator, up to degree ``l_max``.

    By default, this class computes cartesian reals spherical harmonics, i.e.
    :math:`r^l Y^l_m(r)`. You can set ``normalize=True`` to only compute
    :math:`Y^l_m(r)`.

    :param l_max: the maximum degree of the spherical harmonics to be calculated
    :param normalized: whether to compute normalized spherical harmonics or
        cartesian spherical harmonics (default: False)

    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(self, l_max: int, normalized: bool = False):
        self._l_max = l_max

        self._lib = _get_library()

        # intialize both a double precision and a single-precision calculator.
        # we will decide which one to use depending on the dtype of the data
        self._calculator = self._lib.sphericart_new(l_max, normalized)
        self._calculator_f = self._lib.sphericart_new_f(l_max, normalized)

    def __del__(self):
        self._lib.sphericart_delete(self._calculator)
        self._lib.sphericart_delete_f(self._calculator_f)

    def compute(
        self, xyz: np.ndarray, gradients: bool = False
    ) -> (np.ndarray, Optional[np.ndarray]):
        """
        Calculates the spherical harmonics for a set of 3D points, which
        coordinates are in the ``xyz`` array.

        :param xyz: The Cartesian coordinates of the 3D points, as an array with
            shape ``(n_samples, 3)``.
        :param gradients: if ``True``, gradients of the spherical harmonics will
            be calculated and returned in addition to the spherical harmonics.
            ``False`` by default.

        :return: A tuple containing two values:

            * An array of shape ``(n_samples, (l_max+1)**2)`` containing all the
              spherical harmonics up to degree `l_max` in lexicographic order.
              For example, if ``l_max = 2``, The last axis will correspond to
              spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
              1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
            * Either ``None`` if ``gradients=False`` or, if ``gradients=True``,
              an array of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
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
        if gradients:
            dsph = np.empty((n_samples, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)
            dsph_length = n_samples * 3 * (self._l_max + 1) ** 2
        else:
            dsph = None
            dsph_length = 0

        if xyz.dtype == np.float64:
            xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            if gradients:
                dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            else:
                dsph_ptr = None

            self._lib.sphericart_compute_array(
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
            if gradients:
                dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                dsph_ptr = None

            self._lib.sphericart_compute_array(
                self._calculator_f,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                dsph_ptr,
                dsph_length,
            )

        return sph, dsph
