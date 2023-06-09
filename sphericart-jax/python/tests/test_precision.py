import pytest
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax._src.test_util as jtu


import sphericart.jax as sphj


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def test_precision(xyz):
    def compute(xyz):
        sph = sphj.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
        return sph

    xyz_64 = jnp.array(xyz, dtype=jnp.float64)
    xyz_32 = jnp.array(xyz, dtype=jnp.float32)
    assert ((xyz_64 - xyz_32) ** 2).sum() < 1e-8

    sph_64 = compute(xyz_64)
    sph_32 = compute(xyz_32)
    assert ((sph_64 / sph_32 - 1) ** 2).sum() < 1e-5

    dnorm = jax.jit(lambda xyz: compute(xyz).sum())

    assert jnp.allclose(dnorm(sph_64), jnp.array(dnorm(sph_32), dtype=jnp.float64))
