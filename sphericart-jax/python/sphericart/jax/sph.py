import jax
import jax.numpy as jnp
import math
from functools import partial
from jax import core
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir, custom_call
from jax.interpreters import ad

from .dsph import dsph
from .utils import default_layouts


# register the sph primitive
_sph_p = core.Primitive("sph_fwd")
_sph_p.def_impl(partial(xla.apply_primitive, _sph_p))


def sph(xyz, l_max, normalized):
    # Thin wrapper for _sph_p. l_max needs to remain a concrete value,
    # so we pass it as an additional argument
    sph = _sph_p.bind(xyz, l_max, normalized, l_max_c=l_max)
    return sph


def sph_abstract_eval(xyz, l_max, normalized, *, l_max_c):
    # Returns the shape of the output of `sph_fwd`. Needed for jax.jit
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    dtype = xyz.dtype
    out_shape = xyz.shape[:-1] + (sph_size,)
    return ShapedArray(out_shape, dtype)


_sph_p.def_abstract_eval(sph_abstract_eval)


def sph_lowering_cpu(ctx, xyz, l_max, normalized, *, l_max_c):
    # Define the compilation to XLA of the primitive.
    # (`ctx` is a context object)

    # build shapes and dtypes of the inputs and outputs
    xyz_type = ir.RankedTensorType(xyz.type)
    xyz_shape = xyz_type.shape
    dtype = xyz_type.element_type
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    out_shape = xyz_shape[:-1] + [
        sph_size,
    ]
    n_samples = math.prod(xyz_shape[:-1])

    # make sure we dispatch to the correct implementation
    if dtype == ir.F32Type.get():
        op_name = "cpu_sph_f32"
    elif dtype == ir.F64Type.get():
        op_name = "cpu_sph_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
        op_name,
        # Output types
        result_types=[
            mlir.ir.RankedTensorType.get(out_shape, dtype),
        ],
        # inputs to the binded functions
        operands=[
            xyz,
            mlir.ir_constant(l_max_c),
            normalized,
            mlir.ir_constant(n_samples),
        ],
        # Layout specification:
        operand_layouts=default_layouts(xyz_shape, (), (), ()),
        result_layouts=default_layouts(out_shape),
    ).results


mlir.register_lowering(_sph_p, sph_lowering_cpu, platform="cpu")


def sph_p_batch(arg_values, batch_axes, *, l_max_c):
    # Define a batching rule for _sph_p. This is very simple
    # since _sph_p is closed with respect to batching
    res = sph(*arg_values)
    return res, batch_axes[0]


jax.interpreters.batching.primitive_batchers[_sph_p] = sph_p_batch


def sph_jvp(primals, tangents, *, l_max_c):
    # Define the differentiation rule for _sph_p
    sph, d_sph = dsph(*primals)
    return sph, jnp.einsum("...ay, ...a -> ...y", d_sph, tangents[0])


ad.primitive_jvps[_sph_p] = sph_jvp
