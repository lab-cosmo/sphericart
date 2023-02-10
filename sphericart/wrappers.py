import ctypes
import numpy as np


def c_get_prefactors(lib, l_max):
    prefactors = np.empty((l_max+1)*(l_max+2)//2)
    prefactors_ptr = prefactors.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.compute_sph_prefactors(l_max, prefactors_ptr)
    return prefactors


def c_spherical_harmonics(lib, l_max, xyz, prefactors, gradients=False):
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, (l_max+1)**2))
    prefactors_ptr = prefactors.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if gradients:
        dsph = np.empty((n_samples, 3, (l_max+1)**2))
        dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        dsph = None
        dsph_ptr = ctypes.POINTER(ctypes.c_double)()
    lib.cartesian_spherical_harmonics(n_samples, l_max, prefactors_ptr, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph

def c_spherical_harmonics_l0(lib, xyz, gradients=False):    
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, 1))
    xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if gradients:
        dsph = np.empty((n_samples, 3, 1))
        dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        dsph = None
        dsph_ptr = ctypes.POINTER(ctypes.c_double)()
    lib.cartesian_spherical_harmonics_l0(n_samples, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph

def c_spherical_harmonics_l1(lib, xyz, gradients=False):    
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, 4))
    xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if gradients:
        dsph = np.empty((n_samples, 3, 4))
        dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        dsph = None
        dsph_ptr = ctypes.POINTER(ctypes.c_double)()
    lib.cartesian_spherical_harmonics_l1(n_samples, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph

def c_spherical_harmonics_l2(lib, xyz, gradients=False):    
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, 9))
    xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if gradients:
        dsph = np.empty((n_samples, 3, 9))
        dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        dsph = None
        dsph_ptr = ctypes.POINTER(ctypes.c_double)()
    lib.cartesian_spherical_harmonics_l2(n_samples, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph

def c_spherical_harmonics_l3(lib, xyz, gradients=False):    
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, 16))
    xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if gradients:
        dsph = np.empty((n_samples, 3, 16))
        dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        dsph = None
        dsph_ptr = ctypes.POINTER(ctypes.c_double)()
    lib.cartesian_spherical_harmonics_l3(n_samples, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph
