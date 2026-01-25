from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import extend
from jax.core import ShapedArray
from jax.interpreters import ad, mlir, xla

from .ddsph import ddsph


# Register the dsph primitive
_dsph_p = extend.core.Primitive("dsph_fwd")
_dsph_p.multiple_results = True
_dsph_p.def_impl(partial(xla.apply_primitive, _dsph_p))


def dsph(xyz, l_max, normalized):
    """Compute spherical/solid harmonics and their gradients."""
    return _dsph_p.bind(xyz, l_max_c=int(l_max), normalized_c=bool(normalized))


def dsph_abstract_eval(xyz, *, l_max_c, normalized_c):
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    out_shape = xyz.shape[:-1] + (sph_size,)
    dout_shape = xyz.shape[:-1] + (3, sph_size)
    return (ShapedArray(out_shape, xyz.dtype), ShapedArray(dout_shape, xyz.dtype))


_dsph_p.def_abstract_eval(dsph_abstract_eval)


def _op_suffix_from_dtype(dtype):
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def dsph_lowering_cpu(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cpu_"
        + ("dspherical_" if normalized_c else "dsolid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_dsph_p, dsph_lowering_cpu, platform="cpu")


def dsph_lowering_cuda(ctx, xyz, *, l_max_c, normalized_c):
    dtype = np.dtype(ctx.avals_in[0].dtype)
    op_name = (
        "cuda_"
        + ("dspherical_" if normalized_c else "dsolid_")
        + _op_suffix_from_dtype(dtype)
    )
    return jax.ffi.ffi_lowering(op_name)(ctx, xyz, l_max=np.int64(l_max_c))


mlir.register_lowering(_dsph_p, dsph_lowering_cuda, platform="gpu")


def dsph_p_batch(arg_values, batch_axes, *, l_max_c, normalized_c):
    sph_val, dsph_val = dsph(arg_values[0], l_max_c, normalized_c)
    return (sph_val, dsph_val), (batch_axes[0], batch_axes[0])


jax.interpreters.batching.primitive_batchers[_dsph_p] = dsph_p_batch


def dsph_jvp(primals, tangents, *, l_max_c, normalized_c):
    # Use the hessian implementation to get the derivative of dsph.
    sph_val, dsph_val, ddsph_val = ddsph(primals[0], l_max_c, normalized_c)

    # Tangent for sph is contraction of dsph with dx
    sph_t = jnp.einsum("...ay, ...a -> ...y", dsph_val, tangents[0])

    # Tangent for dsph is contraction of ddsph with dx
    # ddsph expected shape: (..., 3, 3, sph_size)
    dsph_t = jnp.einsum("...aby, ...a -> ...by", ddsph_val, tangents[0])

    return (sph_val, dsph_val), (sph_t, dsph_t)


ad.primitive_jvps[_dsph_p] = dsph_jvp
