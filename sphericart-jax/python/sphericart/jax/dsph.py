import jax
import math
from functools import partial
from jax import core
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from .utils import default_layouts


# register the dsph primitive
_dsph_p = core.Primitive("dsph")
_dsph_p.multiple_results = True
_dsph_p.def_impl(partial(xla.apply_primitive, _dsph_p))


def dsph(xyz, l_max, normalized):
    """Define in and out signature and link it to `sph_fwd`"""
    sph, dsph = _dsph_p.bind(xyz, l_max, normalized, l_max_c=l_max)
    return sph, dsph


def dsph_abstract_eval(xyz, l_max, normalized, *, l_max_c):
    """Describe the shape of the output of `sph_fwd`.
    The input `l_max` is an array of size `l_max+1` because
    `l_max` can't be used to compute the shape of the outputs.
    """
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    dtype = xyz.dtype
    sph_shape = xyz.shape[:-1] + (sph_size,)
    dsph_shape = xyz.shape[:-1] + (3, sph_size)
    return ShapedArray(sph_shape, dtype), ShapedArray(dsph_shape, dtype)
_dsph_p.def_abstract_eval(dsph_abstract_eval)


def dsph_lowering_cpu(ctx, xyz, l_max, normalized, *, l_max_c):
    """Define the compilation to XLA of the primitive.

    `ctx` is a context object, the other arguments are just the regular
    inputs.

    """
    # build shapes and dtype of the input and output
    # arguments of the cpp functions
    xyz_type = ir.RankedTensorType(xyz.type)
    xyz_shape = xyz_type.shape
    dtype = xyz_type.element_type
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    sph_shape = xyz_shape[:-1] + [sph_size]
    dsph_shape = xyz_shape[:-1] + [3, sph_size]
    n_samples = math.prod(xyz_shape[:-1])

    if dtype == ir.F32Type.get():
        op_name = "cpu_dsph_f32"
    elif dtype == ir.F64Type.get():
        op_name = "cpu_dsph_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
        op_name,
        # Output types
        out_types=[
            mlir.ir.RankedTensorType.get(sph_shape, dtype),
            mlir.ir.RankedTensorType.get(dsph_shape, dtype),
        ],
        # inputs to the binded functions
        operands=[xyz, mlir.ir_constant(l_max_c), normalized, mlir.ir_constant(n_samples)],
        # Layout specification:
        operand_layouts=default_layouts(xyz_shape, (), (), ()),
        result_layouts=default_layouts(sph_shape, dsph_shape),
    )  # Not sure why this list is necessary here
mlir.register_lowering(_dsph_p, dsph_lowering_cpu, platform="cpu")

def dsph_p_batch(arg_values, batch_axes, *, l_max_c):
    """Computes the batched version of the primitive.
    
    This must be a JAX-traceable function.
    
    Since the multiply_add primitive already operates pointwise on arbitrary
    dimension tensors, to batch it we can use the primitive itself. This works as
    long as both the inputs have the same dimensions and are batched along the
    same axes. The result is batched along the axis that the inputs are batched.
    
    Args:
        vector_arg_values: a tuple of two arguments, each being a tensor of matching
        shape.
        batch_axes: the axes that are being batched. See vmap documentation.
    Returns:
        a tuple of the result, and the result axis that was batched. 
    """
    res = dsph(*arg_values)  # dsph_p is closed w.r.t. batching
    return res, batch_axes[0]
jax.interpreters.batching.primitive_batchers[_dsph_p] = dsph_p_batch
