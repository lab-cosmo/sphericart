import jax
import math
from functools import partial
from jax import core
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir, custom_call
from .utils import default_layouts


# register the dsph primitive
_ddsph_p = core.Primitive("ddsph")
_ddsph_p.multiple_results = True
_ddsph_p.def_impl(partial(xla.apply_primitive, _ddsph_p))


def ddsph(xyz, l_max, normalized):
    sph, dsph, ddsph = _ddsph_p.bind(xyz, l_max, normalized, l_max_c=l_max)
    return sph, dsph, ddsph


def ddsph_abstract_eval(xyz, l_max, normalized, *, l_max_c):
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    dtype = xyz.dtype
    sph_shape = xyz.shape[:-1] + (sph_size,)
    dsph_shape = xyz.shape[:-1] + (3, sph_size)
    ddsph_shape = xyz.shape[:-1] + (3, 3, sph_size)
    return ShapedArray(sph_shape, dtype), ShapedArray(dsph_shape, dtype), ShapedArray(ddsph_shape, dtype)
_ddsph_p.def_abstract_eval(ddsph_abstract_eval)


def ddsph_lowering_cpu(ctx, xyz, l_max, normalized, *, l_max_c):

    xyz_type = ir.RankedTensorType(xyz.type)
    xyz_shape = xyz_type.shape
    dtype = xyz_type.element_type
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    sph_shape = xyz_shape[:-1] + [sph_size]
    dsph_shape = xyz_shape[:-1] + [3, sph_size]
    ddsph_shape = xyz_shape[:-1] + [3, 3, sph_size]
    n_samples = math.prod(xyz_shape[:-1])

    if dtype == ir.F32Type.get():
        op_name = "cpu_ddsph_f32"
    elif dtype == ir.F64Type.get():
        op_name = "cpu_ddsph_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
        op_name,
        result_types=[
            mlir.ir.RankedTensorType.get(sph_shape, dtype),
            mlir.ir.RankedTensorType.get(dsph_shape, dtype),
            mlir.ir.RankedTensorType.get(ddsph_shape, dtype),
        ],
        operands=[xyz, mlir.ir_constant(l_max_c), normalized, mlir.ir_constant(n_samples)],
        operand_layouts=default_layouts(xyz_shape, (), (), ()),
        result_layouts=default_layouts(sph_shape, dsph_shape, ddsph_shape),
    ).results
mlir.register_lowering(_ddsph_p, ddsph_lowering_cpu, platform="cpu")


def ddsph_p_batch(arg_values, batch_axes, *, l_max_c):
    res = ddsph(*arg_values)
    return res, batch_axes[0]
jax.interpreters.batching.primitive_batchers[_ddsph_p] = ddsph_p_batch
