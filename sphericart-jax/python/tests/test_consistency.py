import pytest
import jax
import sphericart.jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def test_consistency(xyz):

    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
            sph = sphericart.jax.spherical_harmonics(l_max=l_max, normalized=normalized, xyz=xyz)

            sph_ref = calculator.compute(np.asarray(xyz))
            np.testing.assert_allclose(sph, sph_ref)
