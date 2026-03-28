import ctypes

import numpy as np


try:
    import cupy as cp
    from cupy import ndarray as cupy_ndarray
except ImportError:
    cp = None

    class cupy_ndarray:  # type: ignore[no-redef]
        pass


def is_array(array):
    return isinstance(array, np.ndarray) or isinstance(array, cupy_ndarray)


def is_cupy_array(array):
    return isinstance(array, cupy_ndarray)


def make_contiguous(array):
    if isinstance(array, np.ndarray):
        return np.ascontiguousarray(array)
    if isinstance(array, cupy_ndarray):
        return cp.ascontiguousarray(array)
    raise TypeError(f"only numpy and cupy arrays are supported, found {type(array)}")


def empty_like(shape, array):
    if isinstance(array, np.ndarray):
        return np.empty(shape, dtype=array.dtype)
    if isinstance(array, cupy_ndarray):
        return cp.empty(shape, dtype=array.dtype)
    raise TypeError(f"only numpy and cupy arrays are supported, found {type(array)}")


def get_pointer(array):
    if array.dtype == np.float32:
        ptr_type = ctypes.POINTER(ctypes.c_float)
    elif array.dtype == np.float64:
        ptr_type = ctypes.POINTER(ctypes.c_double)
    else:
        raise TypeError(
            f"only float32 and float64 arrays are supported, found {array.dtype}"
        )

    if isinstance(array, np.ndarray):
        return array.ctypes.data_as(ptr_type)
    if isinstance(array, cupy_ndarray):
        return ctypes.cast(array.data.ptr, ptr_type)
    raise TypeError(f"only numpy and cupy arrays are supported, found {type(array)}")


def get_cuda_stream(array):
    if not isinstance(array, cupy_ndarray):
        return None
    return ctypes.c_void_p(cp.cuda.get_current_stream().ptr)
