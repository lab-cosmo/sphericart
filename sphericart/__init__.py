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


# Define calls to the wrappers

def get_prefactors(l_max):
    return c_get_prefactors(lib, l_max)

def spherical_harmonics(l_max, xyz, prefactors, gradients=False):
    return c_spherical_harmonics(lib, l_max, xyz, prefactors, gradients)



