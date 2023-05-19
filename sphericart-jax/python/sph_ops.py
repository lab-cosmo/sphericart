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
import jax._src.test_util as jtu

from sphericart import SphericalHarmonics as SphericalHarmonicsCPU

import sys
sys.path.insert(0,'../../build/sphericart-jax/')
import sphericart_jax

from trace_jax import trace

for _name, _value in sphericart_jax.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


# register the SPH primitive
_sph_fwd_p = core.Primitive("sph_fwd")
_sph_fwd_p.multiple_results = True
_sph_fwd_p.def_impl(partial(xla.apply_primitive, _sph_fwd_p))

# @trace("sph_fwd")
def sph_fwd(ls, normalized, xyz):
    """Define in and out signature and link it to `sph_fwd`
    """
    sph, dsph = _sph_fwd_p.bind(ls, normalized, xyz)
    return sph, (dsph, )

# @trace("sph_abstract_eval")
def sph_abstract_eval(ls, normalized, xyz):
    """Describe the shape of the output of `sph_fwd`.
    The input `ls` is an array of size `l_max+1` because
    `l_max` can't be used to compute the shape of the outputs.
    """
    n_sample = xyz.shape[0]
    l_max = ls.shape[0] - 1
    sph_size = (l_max+1)*(l_max+1)
    dtype = xyz.dtype
    shape = (n_sample, sph_size)
    dshape = (n_sample, 3, sph_size)
    return (ShapedArray(shape, dtype), ShapedArray(dshape, dtype))

# @trace("sph_lowering_cpu")
def sph_lowering_cpu(ctx, ls, normalized, xyz):
    """Define the compilation to XLA of the primitive.

    `ctx` is a context object, the other arguments are just the regular
    inputs.

    """
    # build shapes and dtype of the input and output
    # arguments of the cpp functions
    x_type = ir.RankedTensorType(xyz.type)
    x_shape = x_type.shape
    dtype = x_type.element_type
    n_sample = x_shape[0]
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
      # inputs to the binded functions
      operands=[mlir.ir_constant(l_max), normalized,
                    mlir.ir_constant(n_sample), xyz],
      # Layout specification:
      operand_layouts=default_layouts((),(),(),x_shape),
      result_layouts=default_layouts(shape, dshape)
    )
# register the custom xla lowering
mlir.register_lowering(
    _sph_fwd_p,
    sph_lowering_cpu,
    platform="cpu",
)

_sph_fwd_p.def_abstract_eval(sph_abstract_eval)

### define how to compute the gradients backward

# @trace("sph_jvp")
def sph_jvp(l_max, normalized, xyz):
    ls = jnp.arange(l_max+1)
    sph, dsph = sph_fwd(ls, normalized, xyz)
    return sph, dsph

# @trace("sph_vjp")
def sph_vjp(l_max, normalized, args, tangents):
    dsph = args[0]
    out = jnp.einsum('ndl,nl->nd', dsph, tangents)
    return (out,)

@partial(jax.custom_vjp, nondiff_argnums=(0,1))
def spherical_harmonics(l_max, normalized, xyz):
    """Computes the Spherical harmonics and their derivatives within
    the JAX framework. see `sphericart.SphericalHarmonics` for more details.

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics (included)
    normalized : bool
        if the imput vectors have already been normalized
    xyz : jax array [n_sample, 3]
        set of n_sample vectors in 3D

    Returns
    -------
    jax array [n_sample, (l_max+1)**2]
        Spherical harmonics expension of `xyz`
    """
    ls = jnp.arange(l_max+1)
    output, _ = sph_fwd(ls, normalized, xyz)
    return output

spherical_harmonics.defvjp(sph_jvp, sph_vjp)

def func(xyz):
    l_max = 2
    sph = spherical_harmonics(l_max, False, xyz)
    return sph.sum()

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    xyz = 6 * jax.random.normal(key,(4, 3))
    l_max = 5
    ls = jnp.arange(l_max+1)
    normalized = False
    # the lowering works
    sph_t, (dsph_t,) = jit(sph_fwd)(ls, normalized, xyz)

    calculator = SphericalHarmonicsCPU(l_max=l_max, normalized=False)
    sph, grad_sph = calculator.compute(np.asarray(xyz), gradients=True)

    # the computed sph and dsph match the reference impl
    print(np.allclose(sph_t, sph), np.allclose(dsph_t, grad_sph))

    # can jit a function of the SPH
    jfunc = jit(func)
    out = jfunc(xyz)
    # take its gradient
    dout = grad(jfunc)(xyz)
    # the backward gradients match finite difference grads
    jtu.check_grads(jfunc, (xyz,), modes=["bwd"], order=1)
