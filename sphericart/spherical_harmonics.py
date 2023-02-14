from .wrappers import (c_get_prefactors, c_spherical_harmonics, 
c_spherical_harmonics_l0, c_spherical_harmonics_l1, c_spherical_harmonics_l2, c_spherical_harmonics_l3)


class SphericalHarmonics():

    """Docstring?"""

    def __init__(self, l_max):
        self._l_max = l_max
        self._prefactors = c_get_prefactors(l_max)
        if self._l_max==0:
            self._lfun = lambda xyz, gradients : c_spherical_harmonics_l0(xyz, gradients)
        elif self._l_max==1:
            self._lfun = lambda xyz, gradients : c_spherical_harmonics_l1(xyz, gradients) 
        elif self._l_max==2:
            self._lfun = lambda xyz, gradients : c_spherical_harmonics_l2(xyz, gradients)       
        elif self._l_max==3:
            self._lfun = lambda xyz, gradients : c_spherical_harmonics_l3(xyz, gradients)       
        else:
            self._lfun = lambda xyz, gradients : c_spherical_harmonics(self._l_max, xyz, self._prefactors, gradients)       

    def compute(self, xyz, gradients=False):
        return self._lfun(xyz, gradients)
