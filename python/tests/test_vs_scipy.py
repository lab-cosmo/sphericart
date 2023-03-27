import numpy as np
import pytest
import scipy.special

import sphericart


L_MAX = 40
N_SAMPLES = 100


def scipy_real_sph(xyz, l, m):  # noqa E741
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


@pytest.fixture
def xyz():
    np.random.seed(0)
    return np.random.rand(N_SAMPLES, 3)


def test_against_scipy(xyz):
    for l in range(0, L_MAX + 1):  # noqa E741
        for m in range(-l, l + 1):
            sph_scipy_l_m = scipy_real_sph(xyz, l, m)

            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            calculator = sphericart.SphericalHarmonics(l)
            sph_sphericart, _ = calculator.compute(xyz, gradients=False)
            sph_sphericart_l_m = sph_sphericart[:, l * l + l + m] / r**l

            assert np.allclose(sph_scipy_l_m, sph_sphericart_l_m)

            calculator = sphericart.SphericalHarmonics(l, normalized=True)
            sph_sphericart, _ = calculator.compute(xyz, gradients=False)
            sph_normalized_l_m = sph_sphericart[:, l * l + l + m]

            assert np.allclose(sph_scipy_l_m, sph_normalized_l_m)


def test_derivatives(xyz):
    delta = 1e-6
    for normalized in [False, True]:
        calculator = sphericart.SphericalHarmonics(L_MAX, normalized=normalized)
        for alpha in range(3):
            xyz_plus = xyz.copy()
            xyz_plus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
            xyz_minus = xyz.copy()
            xyz_minus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
            sph_plus, _ = calculator.compute(xyz_plus, gradients=False)
            sph_minus, _ = calculator.compute(xyz_minus, gradients=False)
            numerical_derivatives = (sph_plus - sph_minus) / (2.0 * delta)
            sph, analytical_derivatives_all = calculator.compute(xyz, gradients=True)
            analytical_derivatives = analytical_derivatives_all[:, alpha, :]
            assert np.allclose(numerical_derivatives, analytical_derivatives)


if __name__ == "__main__":
    pytest.main([__file__])
