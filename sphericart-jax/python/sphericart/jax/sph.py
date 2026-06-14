from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import extend
from jax.core import ShapedArray
from jax.interpreters import ad, mlir, xla

from .dsph import dsph


# Register the sph primitive
_sph_p = extend.core.Primitive("sph_fwd")
_sph_p.def_impl(partial(xla.apply_primitive, _sph_p))


def sph(xyz, l_max, normalized):
    """Compute spherical (normalized=True) or solid (normalized=False) harmonics."""
    return _sph_p.bind(xyz, l_max_c=int(l_max), normalized_c=bool(normalized))


def sph_abstract_eval(xyz, *, l_max_c, normalized_c):
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    out_shape = xyz.shape[:-1] + (sph_size,)
    return ShapedArray(out_shape, xyz.dtype)


_sph_p.def_abstract_eval(sph_abstract_eval)


def _op_suffix_from_dtype(dtype):
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def sph_lowering_cpu(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cpu_"
        + ("spherical_" if normalized_c else "solid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_sph_p, sph_lowering_cpu, platform="cpu")


def sph_lowering_cuda(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cuda_"
        + ("spherical_" if normalized_c else "solid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_sph_p, sph_lowering_cuda, platform="gpu")


def sph_p_batch(arg_values, batch_axes, *, l_max_c, normalized_c):
    res = sph(arg_values[0], l_max_c, normalized_c)
    return res, batch_axes[0]


jax.interpreters.batching.primitive_batchers[_sph_p] = sph_p_batch


def sph_jvp(primals, tangents, *, l_max_c, normalized_c):
    sph_val, d_sph = dsph(primals[0], l_max_c, normalized_c)
    return sph_val, jnp.einsum("...ay, ...a -> ...y", d_sph, tangents[0])


ad.primitive_jvps[_sph_p] = sph_jvp
