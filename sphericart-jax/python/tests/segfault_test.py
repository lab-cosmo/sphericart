import pytest
import jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax._src.test_util as jtu
import sphericart.jax

def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def compute(xyz):
    sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
    assert jnp.linalg.norm(sph) != 0.0
    return sph.sum()
    

sph = compute(xyz())

print (sph)