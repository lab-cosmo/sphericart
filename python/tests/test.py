import unittest

import numpy as np
import scipy.special

import sphericart

L_MAX = 20
N_SAMPLES = 10


def scipy_real_sph(xyz, l, m):  # noqa E741
    # Scipy spherical harmonics:
    # Note: scipy's theta and phi are the opposite of those in our convention
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    complex_sh_scipy_l_m = scipy.special.sph_harm(m, l, phi, theta)
    complex_sh_scipy_l_negm = scipy.special.sph_harm(-m, l, phi, theta)

    if m > 0:
        sh_scipy_l_m = (
            (complex_sh_scipy_l_negm + (-1) ** m * complex_sh_scipy_l_m) / np.sqrt(2.0)
        ).real
    elif m < 0:
        sh_scipy_l_m = (
            (-complex_sh_scipy_l_m + (-1) ** m * complex_sh_scipy_l_negm) / np.sqrt(2.0)
        ).imag
    else:  # m == 0
        sh_scipy_l_m = complex_sh_scipy_l_m.real

    return sh_scipy_l_m


class TestSphericart(unittest.TestCase):
    def test_against_scipy(self):
        xyz = np.random.rand(N_SAMPLES, 3)

        for l in range(0, L_MAX + 1):  # noqa E741
            for m in range(-l, l + 1):
                sh_scipy_l_m = scipy_real_sph(xyz, l, m)

                x = xyz[:, 0]
                y = xyz[:, 1]
                z = xyz[:, 2]
                r = np.sqrt(x**2 + y**2 + z**2)
                sh_calculator = sphericart.SphericalHarmonics(l)
                sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
                sh_sphericart_l_m = sh_sphericart[:, l * l + l + m] / r**l

                self.assertTrue(np.allclose(sh_scipy_l_m, sh_sphericart_l_m))

                sh_calculator = sphericart.SphericalHarmonics(l, normalized=True)
                sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
                sh_normalized_l_m = sh_sphericart[:, l * l + l + m]

                self.assertTrue(np.allclose(sh_scipy_l_m, sh_normalized_l_m))

    def test_derivatives(self):
        xyz = np.random.rand(N_SAMPLES, 3)

        delta = 1e-6
        for normalized in [False, True]:
            sh_calculator = sphericart.SphericalHarmonics(L_MAX, normalized=normalized)
            for alpha in range(3):
                xyzplus = xyz.copy()
                xyzplus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
                xyzminus = xyz.copy()
                xyzminus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
                shplus, _ = sh_calculator.compute(xyzplus, gradients=False)
                shminus, _ = sh_calculator.compute(xyzminus, gradients=False)
                numerical_derivatives = (shplus - shminus) / (2.0 * delta)
                sh, analytical_derivatives_all = sh_calculator.compute(
                    xyz, gradients=True
                )
                analytical_derivatives = analytical_derivatives_all[:, alpha, :]
                self.assertTrue(
                    np.allclose(numerical_derivatives, analytical_derivatives)
                )


if __name__ == "__main__":
    unittest.main()
