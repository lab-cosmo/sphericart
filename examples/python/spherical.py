import numpy as np
from sphericart import SphericalHarmonics as CartesianSphericalHarmonics


class SphericalHarmonics:
    def __init__(self, l_max):
        self.l_max = l_max
        self.cartesian_spherical_harmonics = CartesianSphericalHarmonics(l_max, normalized=True)

    def compute(self, theta, phi):
        assert theta.shape == phi.shape
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        xyz = np.stack([x, y, z], axis=-1)
        return self.cartesian_spherical_harmonics.compute(xyz)
