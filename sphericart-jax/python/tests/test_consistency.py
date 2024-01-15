import pytest
import jax
import sphericart.jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_consistency(xyz, l_max, normalized):
    calculator = sphericart.SphericalHarmonics(
        l_max=l_max, normalized=normalized
    )
    sph = sphericart.jax.spherical_harmonics(
        l_max=l_max, normalized=normalized, xyz=xyz
    )

    sph_ref = calculator.compute(np.asarray(xyz))
    np.testing.assert_allclose(sph, sph_ref)
