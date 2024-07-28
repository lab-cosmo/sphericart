import numpy as np
from sphericart import SphericalHarmonics as RealSphericalHarmonics


class SphericalHarmonics:
    def __init__(self, l_max):
        self.l_max = l_max
        self.real_spherical_harmonics = RealSphericalHarmonics(l_max, normalized=True)

    def compute(self, xyz):
        if xyz.dtype == np.float32:
            complex_dtype = np.complex64
        elif xyz.dtype == np.float64:
            complex_dtype = np.complex128
        
        real_spherical_harmonics = self.real_spherical_harmonics.compute(xyz)

        one_over_sqrt_2 = 1.0 / np.sqrt(2.0)

        complex_spherical_harmonics = np.zeros((self.l_max + 1) ** 2, dtype=complex_dtype)
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                l_m_index = l ** 2 + l + m
                l_minus_m_index = l ** 2 + l - m
                if m < 0:
                    complex_spherical_harmonics[..., l_m_index] = (
                        one_over_sqrt_2 * (real_spherical_harmonics[..., l_minus_m_index] - 1j * real_spherical_harmonics[..., l_m_index])
                    )
                elif m == 0:
                    complex_spherical_harmonics[..., l_m_index] = real_spherical_harmonics[..., l_m_index]
                else:  # m > 0
                    complex_spherical_harmonics[..., l_m_index] = (
                        (-1)**m * one_over_sqrt_2 * (real_spherical_harmonics[..., l_m_index] + 1j * real_spherical_harmonics[..., l_minus_m_index])
                    )
