import ctypes
import numpy as np


def c_get_prefactors(l_max):
    prefactors = np.empty((l_max+1)*(l_max+2)//2)
    prefactors_ptr = prefactors.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    from .library import lib
    lib.compute_sph_prefactors(l_max, prefactors_ptr)
    return prefactors


def c_spherical_harmonics(l_max, xyz, prefactors, gradients=False):
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
    from .library import lib
    lib.cartesian_spherical_harmonics(n_samples, l_max, prefactors_ptr, xyz_ptr, sph_ptr, dsph_ptr)
    return sph, dsph
