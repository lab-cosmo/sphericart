import ctypes
from .wrappers import c_get_prefactors, c_spherical_harmonics
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


# Define a class which calls the wrappers

class SphericalHarmonics():

    def __init__(self, l_max):
        self.l_max = l_max
        self.prefactors = c_get_prefactors(lib, l_max)

    def compute(self, xyz, gradients=False):
        return c_spherical_harmonics(lib, self.l_max, xyz, self.prefactors, gradients)



