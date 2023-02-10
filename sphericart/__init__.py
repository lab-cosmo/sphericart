import ctypes
from .wrappers import (c_get_prefactors, c_spherical_harmonics, 
c_spherical_harmonics_l0, c_spherical_harmonics_l1, c_spherical_harmonics_l2)
import os


# Load shared object library and define functions upon importing the sphericart module
path_here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.cdll.LoadLibrary(path_here + "/../libsphericart.so")

c_prefactors_fun = lib.compute_sph_prefactors
c_prefactors_fun.restype = None
c_prefactors_fun.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
]

c_spherical_harmonics_fun = lib.cartesian_spherical_harmonics
c_spherical_harmonics_fun.restype = None
c_spherical_harmonics_fun.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

c_spherical_harmonics_l1_fun = lib.cartesian_spherical_harmonics_l1
c_spherical_harmonics_l1_fun.restype = None
c_spherical_harmonics_l1_fun.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

c_spherical_harmonics_l2_fun = lib.cartesian_spherical_harmonics_l2
c_spherical_harmonics_l2_fun.restype = None
c_spherical_harmonics_l2_fun.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

c_spherical_harmonics_l3_fun = lib.cartesian_spherical_harmonics_l1
c_spherical_harmonics_l3_fun.restype = None
c_spherical_harmonics_l3_fun.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
# Define a class which calls the wrappers

class SphericalHarmonics():

    def __init__(self, l_max):
        self._l_max = l_max
        self._prefactors = c_get_prefactors(lib, l_max)
        if self._l_max==0:
            self._lfun = lambda lib,xyz,gradients : c_spherical_harmonics_l0(lib, xyz, gradients)
        elif self._l_max==1:
            self._lfun = lambda lib,xyz,gradients : c_spherical_harmonics_l1(lib, xyz, gradients) 
        elif self._l_max==2:
            self._lfun = lambda lib,xyz,gradients : c_spherical_harmonics_l2(lib, xyz, gradients)       
        elif self._l_max==3:
            self._lfun = lambda lib,xyz,gradients : c_spherical_harmonics_l3(lib, xyz, gradients)       
        else:
            self._lfun = lambda lib,xyz,gradients : c_spherical_harmonics(lib, self._l_max, xyz, self._prefactors, gradients)       

    def compute(self, xyz, gradients=False):
        return self._lfun(lib, xyz, gradients)            
        


