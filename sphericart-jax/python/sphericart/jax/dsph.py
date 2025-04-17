import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import ad, mlir, xla
from jax.interpreters.mlir import custom_call, ir

from .ddsph import ddsph
from .utils import build_sph_descriptor, default_layouts


# This file registers the _dsph_p primitive and defines its implementation,
# as well as some transformation rules. For more information and comments,
# see sph.py

_dsph_p = core.Primitive("dsph")
_dsph_p.multiple_results = True
_dsph_p.def_impl(partial(xla.apply_primitive, _dsph_p))


def dsph(xyz, l_max, normalized):
    sph, dsph = _dsph_p.bind(
        xyz, l_max, normalized, l_max_c=l_max, normalized_c=normalized
    )
    return sph, dsph


def dsph_abstract_eval(xyz, l_max, normalized, *, l_max_c, normalized_c):
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    dtype = xyz.dtype
    sph_shape = xyz.shape[:-1] + (sph_size,)
    dsph_shape = xyz.shape[:-1] + (3, sph_size)
    return core.ShapedArray(sph_shape, dtype), core.ShapedArray(dsph_shape, dtype)


_dsph_p.def_abstract_eval(dsph_abstract_eval)


def dsph_lowering_cpu(ctx, xyz, l_max, normalized, *, l_max_c, normalized_c):
    xyz_type = ir.RankedTensorType(xyz.type)
    xyz_shape = xyz_type.shape
    dtype = xyz_type.element_type
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    sph_shape = xyz_shape[:-1] + [sph_size]
    dsph_shape = xyz_shape[:-1] + [3, sph_size]
    n_samples = math.prod(xyz_shape[:-1])

    op_name = "cpu_d"
    if normalized_c:
        op_name += "spherical_"
    else:
        op_name += "solid_"
    if dtype == ir.F32Type.get():
        op_name += "f32"
    elif dtype == ir.F64Type.get():
        op_name += "f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
        op_name,
        result_types=[
            mlir.ir.RankedTensorType.get(sph_shape, dtype),
            mlir.ir.RankedTensorType.get(dsph_shape, dtype),
        ],
        operands=[
            xyz,
            mlir.ir_constant(l_max_c),
            mlir.ir_constant(n_samples),
        ],
        operand_layouts=default_layouts(xyz_shape, (), ()),
        result_layouts=default_layouts(sph_shape, dsph_shape),
    ).results


mlir.register_lowering(_dsph_p, dsph_lowering_cpu, platform="cpu")


def dsph_lowering_cuda(ctx, xyz, l_max, normalized, *, l_max_c, normalized_c):
    xyz_type = ir.RankedTensorType(xyz.type)
    xyz_shape = xyz_type.shape
    dtype = xyz_type.element_type
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    sph_shape = xyz_shape[:-1] + [sph_size]
    dsph_shape = xyz_shape[:-1] + [3, sph_size]
    n_samples = math.prod(xyz_shape[:-1])

    op_name = "cuda_d"
    if normalized_c:
        op_name += "spherical_"
    else:
        op_name += "solid_"
    if dtype == ir.F32Type.get():
        op_name += "f32"
    elif dtype == ir.F64Type.get():
        op_name += "f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    descriptor = build_sph_descriptor(n_samples, l_max_c)

    return custom_call(
        op_name,
        result_types=[
            mlir.ir.RankedTensorType.get(sph_shape, dtype),
            mlir.ir.RankedTensorType.get(dsph_shape, dtype),
        ],
        operands=[xyz],
        operand_layouts=default_layouts(xyz_shape),
        result_layouts=default_layouts(sph_shape, dsph_shape),
        backend_config=descriptor,
    ).results


mlir.register_lowering(_dsph_p, dsph_lowering_cuda, platform="gpu")


def dsph_p_batch(arg_values, batch_axes, *, l_max_c, normalized_c):
    res = dsph(*arg_values)
    return res, (batch_axes[0], batch_axes[0])


jax.interpreters.batching.primitive_batchers[_dsph_p] = dsph_p_batch


def dsph_jvp(primals, tangents, *, l_max_c, normalized_c):
    sph, d_sph, dd_sph = ddsph(*primals)
    return (sph, d_sph), (
        jnp.einsum("...ay, ...a -> ...y", d_sph, tangents[0]),
        jnp.einsum("...aby, ...a -> ...by", dd_sph, tangents[0]),
    )


ad.primitive_jvps[_dsph_p] = dsph_jvp
