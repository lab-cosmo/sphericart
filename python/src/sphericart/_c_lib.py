import ctypes
import os
import sys

_HERE = os.path.realpath(os.path.dirname(__file__))


def setup_functions(lib):
    lib.sphericart_compute_sph_prefactors.restype = None
    lib.sphericart_compute_sph_prefactors.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]

    lib.sphericart_cartesian_spherical_harmonics.restype = None
    lib.sphericart_cartesian_spherical_harmonics.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    lib.sphericart_normalized_spherical_harmonics.restype = None
    lib.sphericart_normalized_spherical_harmonics.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]


class LibraryFinder(object):
    def __init__(self):
        self._cached_dll = None

    def __call__(self):
        if self._cached_dll is None:
            path = _lib_path()
            self._cached_dll = ctypes.cdll.LoadLibrary(path)
            setup_functions(self._cached_dll)

        return self._cached_dll


def _lib_path():
    if sys.platform.startswith("darwin"):
        windows = False
        name = "libsphericart.dylib"
    elif sys.platform.startswith("linux"):
        windows = False
        name = "libsphericart.so"
    elif sys.platform.startswith("win"):
        windows = True
        name = "sphericart.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")

    path = os.path.join(os.path.join(_HERE, "lib"), name)

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find sphericart shared library at " + path)


def _check_dll(path):
    """
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    """
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but this DLL is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but this DLL is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


_get_library = LibraryFinder()
