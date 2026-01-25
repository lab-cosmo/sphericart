from functools import partial

import numpy as np

import jax
from jax import extend
from jax.core import ShapedArray
from jax.interpreters import mlir, xla


# Register the ddsph primitive
_ddsph_p = extend.core.Primitive("ddsph_fwd")
_ddsph_p.multiple_results = True
_ddsph_p.def_impl(partial(xla.apply_primitive, _ddsph_p))


def ddsph(xyz, l_max, normalized):
    """Compute spherical/solid harmonics, gradients, and Hessians."""
    return _ddsph_p.bind(xyz, l_max_c=int(l_max), normalized_c=bool(normalized))


def ddsph_abstract_eval(xyz, *, l_max_c, normalized_c):
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    out_shape = xyz.shape[:-1] + (sph_size,)
    dout_shape = xyz.shape[:-1] + (3, sph_size)
    ddout_shape = xyz.shape[:-1] + (3, 3, sph_size)
    return (
        ShapedArray(out_shape, xyz.dtype),
        ShapedArray(dout_shape, xyz.dtype),
        ShapedArray(ddout_shape, xyz.dtype),
    )


_ddsph_p.def_abstract_eval(ddsph_abstract_eval)


def _op_suffix_from_dtype(dtype):
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def ddsph_lowering_cpu(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cpu_"
        + ("ddspherical_" if normalized_c else "ddsolid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_ddsph_p, ddsph_lowering_cpu, platform="cpu")


def ddsph_lowering_cuda(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cuda_"
        + ("ddspherical_" if normalized_c else "ddsolid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_ddsph_p, ddsph_lowering_cuda, platform="gpu")


def ddsph_p_batch(arg_values, batch_axes, *, l_max_c, normalized_c):
    sph_val, dsph_val, ddsph_val = ddsph(arg_values[0], l_max_c, normalized_c)
    return (sph_val, dsph_val, ddsph_val), (batch_axes[0], batch_axes[0], batch_axes[0])


jax.interpreters.batching.primitive_batchers[_ddsph_p] = ddsph_p_batch
