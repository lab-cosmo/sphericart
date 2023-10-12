import jax
import jax.numpy as jnp

import math

from functools import partial
from jax import core
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


from .lib import cpu_ops

# register the CPU operation to xla
for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)


# for future work on this code you will want to use the tracing code from
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
# to get a closer look at the call structure of the jax primitives


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


# register the SPH primitive
_sph_p = core.Primitive("sph_fwd")
_sph_p.def_impl(partial(xla.apply_primitive, _sph_p))

# # register the SPH + derivative primitive
# _sph_with_gradients_p = core.Primitive("sph_with_gradients")
# _sph_with_gradients_p.multiple_results = True
# _sph_with_gradients_p.def_impl(partial(xla.apply_primitive, _sph_with_gradients_p))


def sph(xyz, l_max, normalized):
    """Define in and out signature and link it to `sph_fwd`"""
    sph = _sph_p.bind(xyz, l_max, normalized, l_max_c=l_max)
    return sph


# def sph_with_gradients(l_max, normalized, xyz):
#     """Define in and out signature and link it to `sph_fwd`"""
#     sph, dsph = _sph_with_gradients_p.bind(l_max, normalized, xyz)
#     return sph, dsph


def sph_abstract_eval(xyz, l_max, normalized, *, l_max_c):
    """Describe the shape of the output of `sph_fwd`.
    The input `l_max` is an array of size `l_max+1` because
    `l_max` can't be used to compute the shape of the outputs.
    """
    sph_size = (l_max_c + 1) * (l_max_c + 1)
    dtype = xyz.dtype
    out_shape = xyz.shape[:-1] + (sph_size,)
    return ShapedArray(out_shape, dtype)


# def sph_with_gradients_abstract_eval(l_max, normalized, xyz):
#     """Describe the shape of the output of `sph_fwd`.
#     The input `l_max` is an array of size `l_max+1` because
#     `l_max` can't be used to compute the shape of the outputs.
#     """
#     n_sample = xyz.shape[0]
#     sph_size = (l_max + 1) * (l_max + 1)
#     dtype = xyz.dtype
#     shape = (n_sample, sph_size)
#     dshape = (n_sample, 3, sph_size)
#     return (ShapedArray(shape, dtype), ShapedArray(dshape, dtype))


def sph_lowering_cpu(ctx, xyz, l_max, normalized, *, l_max_c):
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
    out_shape = xyz_shape[:-1] + [sph_size,]
    n_samples = math.prod(xyz_shape[:-1])

    if dtype == ir.F32Type.get():
        op_name = "cpu_sph_f32"
    elif dtype == ir.F64Type.get():
        op_name = "cpu_sph_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return [custom_call(
        op_name,
        # Output types
        out_types=[
            mlir.ir.RankedTensorType.get(out_shape, dtype),
        ],
        # inputs to the binded functions
        operands=[xyz, mlir.ir_constant(l_max_c), normalized, mlir.ir_constant(n_samples)],
        # Layout specification:
        operand_layouts=default_layouts(xyz_shape, (), (), ()),
        result_layouts=default_layouts(out_shape),
    )]  # Not sure why this list is necessary here


# def sph_with_gradients_lowering_cpu(ctx, l_max, normalized, xyz):
#     """Define the compilation to XLA of the primitive.

#     `ctx` is a context object, the other arguments are just the regular
#     inputs.

#     """
#     # build shapes and dtype of the input and output
#     # arguments of the cpp functions
#     x_type = ir.RankedTensorType(xyz.type)
#     x_shape = x_type.shape
#     dtype = x_type.element_type
#     n_sample = x_shape[0]
#     sph_size = (l_max + 1) * (l_max + 1)
#     shape = (n_sample, sph_size)
#     dshape = (n_sample, 3, sph_size)

#     if dtype == ir.F32Type.get():
#         op_name = "cpu_sph_with_gradients_f32"
#     elif dtype == ir.F64Type.get():
#         op_name = "cpu_sph_with_gradients_f64"
#     else:
#         raise NotImplementedError(f"Unsupported dtype {dtype}")

#     return custom_call(
#         op_name,
#         # Output types
#         out_types=[
#             mlir.ir.RankedTensorType.get(shape, dtype),
#             mlir.ir.RankedTensorType.get(dshape, dtype),
#         ],
#         # inputs to the binded functions
#         operands=[mlir.ir_constant(l_max), normalized, mlir.ir_constant(n_sample), xyz],
#         # Layout specification:
#         operand_layouts=default_layouts((), (), (), x_shape),
#         result_layouts=default_layouts(shape, dshape),
#     )

# register the custom xla lowering
mlir.register_lowering(
    _sph_p,
    sph_lowering_cpu,
    platform="cpu",
)

# # register the custom xla lowering
# mlir.register_lowering(
#     _sph_with_gradients_p,
#     sph_with_gradients_lowering_cpu,
#     platform="cpu",
# )

_sph_p.def_abstract_eval(sph_abstract_eval)
# _sph_with_gradients_p.def_abstract_eval(sph_with_gradients_abstract_eval)


### define how to compute the gradients backward
# to be able to use dsph in sph_vjp without recomputing it is returned here
# the drawback is that only backward diff. is possible with the current code
# SUGGESTION change the bindings/Cpp API to be able to retrive the gradients
# at a later stage
# def sph_jvp(l_max, normalized, xyz):
#     sph, dsph = sph_with_gradients(l_max, normalized, xyz)
#     return sph, dsph

# def sph_vjp(l_max, normalized, args, tangents):
#     xyz = args[0]
#     _, dsph = sph_with_gradients(l_max, normalized, xyz)
#     out = jnp.einsum("ndl, nl -> nd", dsph, tangents)
#     return (out,)

def sph_p_batch(arg_values, batch_axes, *, l_max_c):
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
    res = sph(*arg_values)  # sph_p is closed w.r.t. batching
    return res, batch_axes[0]

jax.interpreters.batching.primitive_batchers[_sph_p] = sph_p_batch


def spherical_harmonics(xyz, l_max, normalized):  # TODO: CHANGE xyz to first argument everywhere
    """Computes the Spherical harmonics and their derivatives within
    the JAX framework. See :py:class:`sphericart.SphericalHarmonics` for more details.

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics (included)
    normalized : bool
        should we compute cartesian (``normalized=False``) or normalized spherical harmonics
    xyz : jax array [n_sample, 3]
        set of n_sample vectors in 3D

    Returns
    -------
    jax array [n_sample, (l_max+1)**2]
        Spherical harmonics expension of `xyz`
    """
    if xyz.shape[-1] != 3: raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized)
    return output


# spherical_harmonics.defvjp(sph_jvp, sph_vjp)
