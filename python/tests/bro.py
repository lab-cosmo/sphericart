import numpy as np
import sphericart 

np.random.seed(0)
xyz = np.random.rand(50, 3)

delta = 1e-4
for normalized in [False, True]:
    calculator = sphericart.SphericalHarmonics(8, normalized=normalized)
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
            numerical_second_derivatives = (sph_plus_plus - sph_minus_plus - sph_plus_minus + sph_minus_minus) / (4.0 * delta**2)
            _, _, analytical_second_derivatives_all = calculator.compute_with_hessians(xyz)
            analytical_second_derivatives = analytical_second_derivatives_all[:, alpha, beta, :]
            assert np.allclose(numerical_second_derivatives, analytical_second_derivatives, rtol=1e-4, atol=1e-5)
