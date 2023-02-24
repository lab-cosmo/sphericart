from .wrappers import c_get_prefactors, c_spherical_harmonics


class SphericalHarmonics:
    """
    The user-facing spherical harmonics class.

    Upon initialization, this class returns a calculator. The ``l_max`` value is
    used to initialize the calculator, and, in particular, it is used to
    calculate the relevant prefactors for the calculation of the spherical
    harmonics.

    :param l_max:
            the maximum degree of the spherical harmonics to be calculated
    :type l_max: int

    :param normalized:
            whether to compute normalized spherical harmonics (default: False)
    :type normalized: bool

    :return: a calculator, in the form of a SphericalHarmonics object
    """

    def __init__(self, l_max, normalized=False):
        self._l_max = l_max
        self._prefactors = c_get_prefactors(l_max)
        self._normalized = normalized

    def __del__(self):
        # TODO: free the prefactors
        pass

    def compute(self, xyz, gradients=False):
        """
        Calculates the spherical harmonics.

        More specifically, it computes the scaled spherical harmonics :math:`r^l
        Y_l^m` from the Cartesian coordinates of a set of 3D points.

        :param xyz: The Cartesian coordinates of the 3D points, as an array with
            shape ``(n_samples, 3)``.
        :type xyz: numpy.ndarray
        :param gradients: if ``True``, gradients of the spherical harmonics will
            be calculated and returned in addition to the spherical harmonics.
            ``False`` by default.
        :type gradients: bool

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
              the x, y, and z derivatives, respectively.

        :rtype: tuple(``numpy.ndarray``, ``numpy.ndarray`` or ``None``)

        """

        return c_spherical_harmonics(
            self._l_max, xyz, self._prefactors, gradients, self._normalized
        )
