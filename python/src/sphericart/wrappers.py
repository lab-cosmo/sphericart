import ctypes

import numpy as np

from ._c_lib import _get_library


def c_sph_new(l_max, normalized):
    lib = _get_library()
    return [
        lib.sphericart_new(l_max, 1 if normalized else 0),
        lib.sphericart_new_f(l_max, 1 if normalized else 0),
    ]


def c_sph_delete(sph_pointers):
    lib = _get_library()
    lib.sphericart_delete(sph_pointers[0])
    lib.sphericart_delete(sph_pointers[1])


def c_sph_compute(sph_pointers, l_max, xyz, gradients=False, normalized=False):
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, (l_max + 1) ** 2), dtype=xyz.dtype)
    if gradients:
        dsph = np.empty((n_samples, 3, (l_max + 1) ** 2), dtype=xyz.dtype)
    else:
        dsph = None

    lib = _get_library()

    if xyz.dtype == np.float64:
        xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if gradients:
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            dsph_ptr = ctypes.POINTER(ctypes.c_double)()
        lib.sphericart_compute_array(
            sph_pointers[0], n_samples, xyz_ptr, sph_ptr, dsph_ptr
        )
    elif xyz.dtype == np.float32:
        xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sph_ptr = sph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if gradients:
            dsph_ptr = dsph.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            dsph_ptr = ctypes.POINTER(ctypes.c_float)()
        lib.sphericart_compute_array(
            sph_pointers[1], n_samples, xyz_ptr, sph_ptr, dsph_ptr
        )
    else:
        raise ValueError("Can only compute for float32 and float64")
    return sph, dsph
