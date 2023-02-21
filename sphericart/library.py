import ctypes
import os
global lib


# Load shared object library and define functions upon importing the sphericart module
path_here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.cdll.LoadLibrary(path_here + "/lib/libsphericart.so")

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

c_spherical_harmonics_fun = lib.normalized_cartesian_spherical_harmonics
c_spherical_harmonics_fun.restype = None
c_spherical_harmonics_fun.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
