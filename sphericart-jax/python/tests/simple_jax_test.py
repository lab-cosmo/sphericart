import jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from sphericart.jax.spherical_harmonics import spherical_harmonics

def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))

def compute(xyz):
    sph = spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
    assert jnp.linalg.norm(sph) != 0.0
    return sph

# xyzs = jax.device_put(xyz(), device=jax.devices('gpu')[0]) 

sph = compute(xyz())

print (sph)
print ("sum sph:", sph.sum())
print ("cuda jax successful")
