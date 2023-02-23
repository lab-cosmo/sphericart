import numpy as np
import scipy.special

import sphericart


def test_against_scipy(xyz: np.ndarray, l: int, m: int):
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

    sh_calculator = sphericart.SphericalHarmonics(l)
    sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
    sh_sphericart_l_m = sh_sphericart[:, l * l + l + m] / r**l

    assert np.allclose(
        sh_scipy_l_m, sh_sphericart_l_m
    ), f"SPH value failed for l={l}, m={m}"

    sh_calculator = sphericart.SphericalHarmonics(l, normalized=True)
    sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
    sh_normalized_l_m = sh_sphericart[:, l * l + l + m]

    assert np.allclose(
        sh_scipy_l_m, sh_normalized_l_m
    ), f"normalized value failed for l={l}, m={m}"




n_samples = 10
l_max = 20
xyz = np.random.rand(n_samples, 3)
for l in range(0, l_max + 1):
    for m in range(-l, l + 1):
        test_against_scipy(xyz, l, m)

print("Spherical harmonics tests passed successfully!")

delta = 1e-6
for normalized in [False, True]:
    sh_calculator = sphericart.SphericalHarmonics(l_max, normalized=normalized)
    for alpha in range(3):
        xyzplus = xyz.copy()
        xyzplus[:, alpha] += delta * np.ones_like(xyz[:, alpha])
        xyzminus = xyz.copy()
        xyzminus[:, alpha] -= delta * np.ones_like(xyz[:, alpha])
        shplus, _ = sh_calculator.compute(xyzplus, gradients=False)
        shminus, _ = sh_calculator.compute(xyzminus, gradients=False)
        numerical_derivatives = (shplus - shminus) / (2.0 * delta)
        sh, analytical_derivatives_all = sh_calculator.compute(xyz, gradients=True)
        analytical_derivatives = analytical_derivatives_all[:, alpha, :]
        print("analytical,", alpha,  analytical_derivatives[:2, :2] )
        print("numerical,", alpha,  numerical_derivatives[:2, :2] )
        assert np.allclose(numerical_derivatives, 
               analytical_derivatives), f"derivative failed for normalize={normalized})"

print("Derivative tests passed successfully!")
