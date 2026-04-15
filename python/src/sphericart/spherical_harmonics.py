from typing import Tuple

import numpy as np

from . import _dispatch
from ._c_lib import _get_library


class _BaseHarmonics:
    _cpu_prefix = ""
    _cuda_prefix = ""

    def __init__(self, l_max: int):
        self._l_max = l_max
        self._lib = _get_library()

        self._calculator = getattr(self._lib, f"{self._cpu_prefix}_new")(l_max)
        self._calculator_f = getattr(self._lib, f"{self._cpu_prefix}_new_f")(l_max)
        self._calculator_cuda = None
        self._calculator_cuda_f = None

        self._omp_num_threads = getattr(
            self._lib, f"{self._cpu_prefix}_omp_num_threads"
        )(self._calculator)

    def __del__(self):
        self._delete_calculator(f"{self._cpu_prefix}_delete", "_calculator")
        self._delete_calculator(f"{self._cpu_prefix}_delete_f", "_calculator_f")
        self._delete_calculator(f"{self._cuda_prefix}_delete", "_calculator_cuda")
        self._delete_calculator(f"{self._cuda_prefix}_delete_f", "_calculator_cuda_f")

    def _delete_calculator(self, function_name: str, attribute: str):
        calculator = getattr(self, attribute, None)
        if calculator is not None:
            getattr(self._lib, function_name)(calculator)
            setattr(self, attribute, None)

    def _validate_xyz(self, xyz):
        if self._calculator is None or self._calculator_f is None:
            raise ValueError("can not use a deleted calculator")

        if not _dispatch.is_array(xyz):
            raise TypeError("xyz must be a numpy or cupy array")

        if xyz.dtype not in (np.float32, np.float64):
            raise TypeError("xyz must be an array of 32 or 64-bit floats")

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz array must be a `N x 3` array")

        return _dispatch.make_contiguous(xyz)

    def _ensure_cuda_calculators(self):
        if self._calculator_cuda is not None and self._calculator_cuda_f is not None:
            return

        self._calculator_cuda = getattr(self._lib, f"{self._cuda_prefix}_new")(
            self._l_max
        )
        self._calculator_cuda_f = getattr(self._lib, f"{self._cuda_prefix}_new_f")(
            self._l_max
        )

        if self._calculator_cuda and self._calculator_cuda_f:
            return

        self._delete_calculator(f"{self._cuda_prefix}_delete", "_calculator_cuda")
        self._delete_calculator(f"{self._cuda_prefix}_delete_f", "_calculator_cuda_f")
        raise RuntimeError(
            "failed to initialize CUDA calculators; ensure sphericart was built "
            "with CUDA support and the CUDA runtime libraries are available"
        )

    def _empty_output(self, xyz, *shape):
        return _dispatch.empty_like(shape, xyz)

    def _compute_impl(self, xyz, gradients: bool = False, hessians: bool = False):
        xyz = self._validate_xyz(xyz)

        n_samples = xyz.shape[0]
        lmtotal = (self._l_max + 1) ** 2

        sph = self._empty_output(xyz, n_samples, lmtotal)
        dsph = self._empty_output(xyz, n_samples, 3, lmtotal) if gradients else None
        ddsph = self._empty_output(xyz, n_samples, 3, 3, lmtotal) if hessians else None

        if _dispatch.is_cupy_array(xyz):
            self._ensure_cuda_calculators()
            self._compute_cupy(xyz, sph, dsph, ddsph)
        else:
            self._compute_numpy(xyz, sph, dsph, ddsph)

        if hessians:
            return sph, dsph, ddsph
        if gradients:
            return sph, dsph
        return sph

    def _compute_numpy(self, xyz, sph, dsph, ddsph):
        xyz_ptr = _dispatch.get_pointer(xyz)
        sph_ptr = _dispatch.get_pointer(sph)
        xyz_length = xyz.shape[0] * 3
        sph_length = sph.shape[0] * sph.shape[1]

        suffix = "" if xyz.dtype == np.float64 else "_f"
        calculator = self._calculator if xyz.dtype == np.float64 else self._calculator_f

        if ddsph is not None:
            function = getattr(
                self._lib, f"{self._cpu_prefix}_compute_array_with_hessians{suffix}"
            )
            function(
                calculator,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                _dispatch.get_pointer(dsph),
                dsph.shape[0] * dsph.shape[1] * dsph.shape[2],
                _dispatch.get_pointer(ddsph),
                ddsph.shape[0] * ddsph.shape[1] * ddsph.shape[2] * ddsph.shape[3],
            )
        elif dsph is not None:
            function = getattr(
                self._lib, f"{self._cpu_prefix}_compute_array_with_gradients{suffix}"
            )
            function(
                calculator,
                xyz_ptr,
                xyz_length,
                sph_ptr,
                sph_length,
                _dispatch.get_pointer(dsph),
                dsph.shape[0] * dsph.shape[1] * dsph.shape[2],
            )
        else:
            function = getattr(self._lib, f"{self._cpu_prefix}_compute_array{suffix}")
            function(calculator, xyz_ptr, xyz_length, sph_ptr, sph_length)

    def _compute_cupy(self, xyz, sph, dsph, ddsph):
        xyz_ptr = _dispatch.get_pointer(xyz)
        sph_ptr = _dispatch.get_pointer(sph)
        stream = _dispatch.get_cuda_stream(xyz)
        suffix = "" if xyz.dtype == np.float64 else "_f"
        calculator = (
            self._calculator_cuda
            if xyz.dtype == np.float64
            else self._calculator_cuda_f
        )

        if ddsph is not None:
            function = getattr(
                self._lib, f"{self._cuda_prefix}_compute_array_with_hessians{suffix}"
            )
            function(
                calculator,
                xyz_ptr,
                xyz.shape[0],
                sph_ptr,
                _dispatch.get_pointer(dsph),
                _dispatch.get_pointer(ddsph),
                stream,
            )
        elif dsph is not None:
            function = getattr(
                self._lib, f"{self._cuda_prefix}_compute_array_with_gradients{suffix}"
            )
            function(
                calculator,
                xyz_ptr,
                xyz.shape[0],
                sph_ptr,
                _dispatch.get_pointer(dsph),
                stream,
            )
        else:
            function = getattr(self._lib, f"{self._cuda_prefix}_compute_array{suffix}")
            function(calculator, xyz_ptr, xyz.shape[0], sph_ptr, stream)


class SphericalHarmonics(_BaseHarmonics):
    """
    Spherical harmonics calculator, which computes the real spherical harmonics
    :math:`Y^m_l` up to degree ``l_max``.

    The calculator accepts either NumPy arrays on CPU or CuPy arrays on CUDA.
    """

    _cpu_prefix = "sphericart_spherical_harmonics"
    _cuda_prefix = "sphericart_cuda_spherical_harmonics"

    def compute(self, xyz):
        """Calculate spherical harmonics for an ``(n_samples, 3)`` array."""
        return self._compute_impl(xyz)

    def compute_with_gradients(self, xyz) -> Tuple[object, object]:
        """Calculate spherical harmonics and Cartesian gradients."""
        return self._compute_impl(xyz, gradients=True)

    def compute_with_hessians(self, xyz) -> Tuple[object, object, object]:
        """Calculate spherical harmonics, gradients, and Hessians."""
        return self._compute_impl(xyz, gradients=True, hessians=True)


class SolidHarmonics(_BaseHarmonics):
    """
    Solid harmonics calculator, up to degree ``l_max``.

    The calculator accepts either NumPy arrays on CPU or CuPy arrays on CUDA.
    """

    _cpu_prefix = "sphericart_solid_harmonics"
    _cuda_prefix = "sphericart_cuda_solid_harmonics"

    def compute(self, xyz):
        """Calculate solid harmonics for an ``(n_samples, 3)`` array."""
        return self._compute_impl(xyz)

    def compute_with_gradients(self, xyz) -> Tuple[object, object]:
        """Calculate solid harmonics and Cartesian gradients."""
        return self._compute_impl(xyz, gradients=True)

    def compute_with_hessians(self, xyz) -> Tuple[object, object, object]:
        """Calculate solid harmonics, gradients, and Hessians."""
        return self._compute_impl(xyz, gradients=True, hessians=True)
