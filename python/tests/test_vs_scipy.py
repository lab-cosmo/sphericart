import numpy as np
import pytest
import scipy.special

import sphericart


L_MAX = 15
N_SAMPLES = 100


def scipy_real_sph(xyz, l, m):  # noqa E741
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    complex_sh_scipy_l_m = scipy.special.sph_harm_y(l, m, theta, phi)
    complex_sh_scipy_l_negm = scipy.special.sph_harm_y(l, -m, theta, phi)

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
            calculator = sphericart.SolidHarmonics(l)
            sph_sphericart = calculator.compute(xyz)
            sph_sphericart_l_m = sph_sphericart[:, l * l + l + m] / r**l

            assert np.allclose(sph_scipy_l_m, sph_sphericart_l_m)

            calculator = sphericart.SphericalHarmonics(l)
            sph_sphericart = calculator.compute(xyz)
            sph_normalized_l_m = sph_sphericart[:, l * l + l + m]

            assert np.allclose(sph_scipy_l_m, sph_normalized_l_m)


def test_derivatives(xyz):
    delta = 1e-6
    for calculator in [
        sphericart.SphericalHarmonics(L_MAX),
        sphericart.SolidHarmonics(L_MAX),
    ]:
        for alpha in range(3):
            xyz_plus = xyz.copy()
            xyz_plus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
            xyz_minus = xyz.copy()
            xyz_minus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
            sph_plus = calculator.compute(xyz_plus)
            sph_minus = calculator.compute(xyz_minus)
            numerical_derivatives = (sph_plus - sph_minus) / (2.0 * delta)
            _, analytical_derivatives_all = calculator.compute_with_gradients(xyz)
            analytical_derivatives = analytical_derivatives_all[:, alpha, :]
            assert np.allclose(numerical_derivatives, analytical_derivatives)


def test_second_derivatives(xyz):
    delta = 1e-5
    for calculator in [
        sphericart.SphericalHarmonics(L_MAX),
        sphericart.SolidHarmonics(L_MAX),
    ]:
        for alpha in range(3):
            for beta in range(3):
                xyz_plus_plus = xyz.copy()
                xyz_plus_plus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
                xyz_plus_plus[:, beta] += delta * np.ones_like(xyz[:, beta])
                xyz_plus_minus = xyz.copy()
                xyz_plus_minus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
                xyz_plus_minus[:, beta] -= delta * np.ones_like(xyz[:, beta])
                xyz_minus_plus = xyz.copy()
                xyz_minus_plus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
                xyz_minus_plus[:, beta] += delta * np.ones_like(xyz[:, beta])
                xyz_minus_minus = xyz.copy()
                xyz_minus_minus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
                xyz_minus_minus[:, beta] -= delta * np.ones_like(xyz[:, beta])
                sph_plus_plus = calculator.compute(xyz_plus_plus)
                sph_plus_minus = calculator.compute(xyz_plus_minus)
                sph_minus_plus = calculator.compute(xyz_minus_plus)
                sph_minus_minus = calculator.compute(xyz_minus_minus)
                numerical_second_derivatives = (
                    sph_plus_plus - sph_minus_plus - sph_plus_minus + sph_minus_minus
                ) / (4.0 * delta**2)
                (
                    _,
                    _,
                    analytical_second_derivatives_all,
                ) = calculator.compute_with_hessians(xyz)
                analytical_second_derivatives = analytical_second_derivatives_all[
                    :, alpha, beta, :
                ]
                # Lower the tolerances as numerical second derivatives are imprecise:
                print(numerical_second_derivatives - analytical_second_derivatives)
                assert np.allclose(
                    numerical_second_derivatives,
                    analytical_second_derivatives,
                    rtol=1e-4,
                    atol=1e-3,
                )


if __name__ == "__main__":
    pytest.main([__file__])
