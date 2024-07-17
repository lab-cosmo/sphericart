import jax.numpy as jnp
import pytest

import sphericart.jax


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("l_max", [0, 3, 7, 10, 20, 50])
def test_no_points(l_max, normalized):
    xyz = jnp.empty((0, 3))

    sph = sphericart.jax.spherical_harmonics(
        l_max=l_max, normalized=normalized, xyz=xyz
    )
    assert sph.shape == (0, l_max * l_max + 2 * l_max + 1)
