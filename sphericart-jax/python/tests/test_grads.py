import numpy as np
import jax
import jax.numpy as jnp
import pytest
import sphericart.jax
from pure_jax_sph import pure_jax_spherical_harmonics

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (9, 3))


def test_gradgrad(xyz, l_max):
    sum_sph = lambda x, l_max, normalized: jnp.sum(
        sphericart.jax.spherical_harmonics(x, l_max, normalized)
    )
    pure_jax_sum_sph = lambda x, l_max: jnp.sum(pure_jax_spherical_harmonics(x, l_max))
    sum_grad_sph = lambda x, l_max, normalized: jnp.sum(
        jax.grad(sum_sph)(x, l_max, normalized)
    )
    pure_jax_sum_grad_sph = lambda x, l_max: jnp.sum(
        jax.grad(pure_jax_sum_sph)(x, l_max)
    )
    gradgrad_sph = jax.grad(sum_grad_sph)
    pure_jax_gradgrad_sph = jax.grad(pure_jax_sum_grad_sph)
    sph = gradgrad_sph(xyz, l_max, normalized=True)
    sph_pure_jax = pure_jax_gradgrad_sph(xyz, l_max)
    
    print (sph)
    print (sph_pure_jax)
    
    assert jnp.allclose(sph, sph_pure_jax, atol=1e-4, rtol=3e-4)
    
    print ("assertion passed")
    
xyzs = xyz()
print ("xyz device",xyzs.devices())

test_gradgrad(xyzs, 4)