import jax
import jax.numpy as jnp
import pytest

import sphericart.jax


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def test_precision(xyz):
    jax.config.update("jax_enable_x64", True)

    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(xyz, l_max=4, normalized=False)
        return sph

    xyz_64 = jnp.array(xyz, dtype=jnp.float64)
    xyz_32 = jnp.array(xyz, dtype=jnp.float32)
    assert ((xyz_64 - xyz_32) ** 2).sum() < 1e-8

    sph_64 = compute(xyz_64)
    sph_32 = compute(xyz_32)
    assert ((sph_64 / sph_32 - 1) ** 2).sum() < 1e-5

    dnorm = jax.jit(lambda xyz: compute(xyz).sum())

    assert jnp.allclose(dnorm(xyz_64), jnp.array(dnorm(xyz_32), dtype=jnp.float64))

    jax.config.update("jax_enable_x64", False)  # avoid affecting other tests
