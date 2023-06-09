import pytest
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sphericart.jax as sphj
from sphericart import SphericalHarmonics as SphericalHarmonicsCPU


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def test_consistency(xyz):
    xyz = jnp.array(xyz, dtype=jnp.float64)

    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            calculator = SphericalHarmonicsCPU(l_max=l_max, normalized=normalized)
            sph = sphj.spherical_harmonics(l_max=l_max, normalized=normalized, xyz=xyz)

            sph_ref = calculator.compute(np.asarray(xyz))
            np.testing.assert_allclose(sph, sph_ref)
