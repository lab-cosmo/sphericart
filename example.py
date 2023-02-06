import numpy as np
import scipy as sp
import ctypes

lib = ctypes.cdll.LoadLibrary("./libsphericart.so")

c_prefactors = lib.compute_sph_prefactors
c_prefactors.restype = None
c_prefactors.argtypes = [
    ctypes.c_uint,
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

c_spherical_harmonics = lib.cartesian_spherical_harmonics_naive
c_spherical_harmonics.restype = None
c_spherical_harmonics.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

def get_prefactors(l_max):
    prefactors = np.empty((l_max+1)*(l_max+2)//2)
    c_prefactors(l_max, prefactors)
    return prefactors

def spherical_harmonics(l_max, xyz, prefactors, gradients=False):
    n_samples = xyz.shape[0]
    sph = np.empty((n_samples, (l_max+1)**2))
    if gradients:
        pass  # allocate gradients
    else:
        dsph = np.empty((0,))  # Need to find a way to create a numpy array that points to null. Not sure if this is correct.
    c_spherical_harmonics(n_samples, l_max, prefactors, xyz, sph, dsph)
    return sph


def test_against_scipy(xyz: np.ndarray, l: int, m: int):

    # Scipy spherical harmonics:
    # Note: scipy's theta and phi are the opposite of those in our convention
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    complex_sh_scipy_l_m = sp.special.sph_harm(m, l, phi, theta)
    complex_sh_scipy_l_negm = sp.special.sph_harm(-m, l, phi, theta)

    if m > 0:
        sh_scipy_l_m = ((complex_sh_scipy_l_negm+(-1)**m*complex_sh_scipy_l_m)/np.sqrt(2.0)).real
    elif m < 0:
        sh_scipy_l_m = ((-complex_sh_scipy_l_m+(-1)**m*complex_sh_scipy_l_negm)/np.sqrt(2.0)).imag
    else: # m == 0
        sh_scipy_l_m = complex_sh_scipy_l_m.real

    prefactors = get_prefactors(l_max)
    sh_sphericart = spherical_harmonics(l_max, xyz, prefactors, gradients=False)
    sh_sphericart_l_m = sh_sphericart[:, l*l+l+m] / r**l

    assert np.allclose(sh_scipy_l_m, sh_sphericart_l_m), f"assertion failed for l={l}, m={m}"


n_samples = 10
l_max = 5
xyz = np.random.rand(n_samples, 3)
for l in range(0, l_max+1):
    for m in range(-l, l+1):
        test_against_scipy(xyz, l, m)

print("Assertions passed successfully!")
