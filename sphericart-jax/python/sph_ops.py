from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

import torch
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import numpy as np

from functools import partial, reduce
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jax.sharding import Mesh, PartitionSpec
from jaxlib.hlo_helpers import custom_call

import sys
sys.path.insert(0,'../../build/sphericart-jax/')
import sphericart_jax
# from sphericart import SphericalHarmonics as SphericalHarmonicsCPU

from trace_jax import trace


for _name, _value in sphericart_jax.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]



# def sph_impl(l_max, normalized, xyz):
#     calc = sphericart_.SphericalHarmonics(l_max=l_max, normalized=False)
#     sph = calc.compute(xyz, gradients=False)
#     return sph


_sph_fwd_p = core.Primitive("sph_fwd")
_sph_fwd_p.multiple_results = True
_sph_fwd_p.def_impl(partial(xla.apply_primitive, _sph_fwd_p))

@trace("sph_fwd")
def sph_fwd(ls, normalized, xyz):
    output, invvar = _sph_fwd_p.bind(ls, normalized, xyz)
    return output, (invvar, xyz)

@trace("sph_abstract_eval")
def sph_abstract_eval(ls, normalized, xyz):
    n_sample = xyz.shape[0]
    l_max = ls.shape[0] - 1
    sph_size = (l_max+1)*(l_max+1)
    dtype = xyz.dtype
    shape = (n_sample, sph_size)
    dshape = (n_sample, 3, sph_size)
    return (ShapedArray(shape, dtype), ShapedArray(dshape, dtype))

@trace("sph_lowering_cpu")
def sph_lowering_cpu(ctx, ls, normalized, xyz):
    """The compilation to XLA of the primitive.

    Given an XlaBuilder and XlaOps for each argument, return the XlaOp for the
    result of the function.

    Does not need to be a JAX-traceable function.
    """
    x_type = ir.RankedTensorType(xyz.type)
    x_shape = x_type.shape
    dtype = x_type.element_type
    # np_dtype = np.dtype(x_type.element_type)
    n_sample = x_shape[0]
    # print("@@@@@@@",dtype, type(dtype), dtype==ir.F32Type.get(), dtype==ir.F64Type.get())
    ls_type = ir.RankedTensorType(ls.type)
    ls_shape = ls_type.shape
    l_max = ls_shape[0] - 1
    sph_size = (l_max+1)*(l_max+1)
    shape = (n_sample, sph_size)
    dshape = (n_sample, 3, sph_size)

    if dtype == ir.F32Type.get():
        op_name = "cpu_sph_with_gradients_f32"
    elif dtype == ir.F64Type.get():
        op_name = "cpu_sph_with_gradients_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
      op_name,
      # Output types
      out_types=[
          mlir.ir.RankedTensorType.get(shape, dtype),
          mlir.ir.RankedTensorType.get(dshape, dtype),
      ],
      # The inputs:
      operands=[l_max, normalized, xyz],
      # Layout specification:
      operand_layouts=default_layouts((),(),x_shape),
      result_layouts=default_layouts(shape, dshape)
    )

mlir.register_lowering(
    _sph_fwd_p,
    sph_lowering_cpu,
    platform="cpu",
)

# xla_client.ops.CustomCall
# _sph_fwd_p.def_impl(sph_impl)
_sph_fwd_p.def_abstract_eval(sph_abstract_eval)



_sph_bwd_p = core.Primitive("sph_bwd")
_sph_bwd_p.multiple_results = True
_sph_bwd_p.def_impl(partial(xla.apply_primitive, _sph_bwd_p))

@trace("sph_bwd")
def sph_bwd(normalized, ls, res, g):
    invvar, xyz = res
    grad_input, grad_weight, part_grad = _sph_bwd_p.bind(
        g, invvar, xyz, ls
    )
    return grad_input, grad_weight





if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    xyz = 6 * jax.random.normal(key,(4, 3))
    l_max = 2
    ls = jnp.arange(l_max+1)
    normalized = False
    out = sph_fwd(ls, normalized, xyz)

    # calculator = sphericart_torch.SphericalHarmonics(l_max=l_max, normalized=False)
    # sph, grad_sph = calculator.compute(xyz, gradients=True)
    # print(sph.shape, (l_max+1) ** 2)