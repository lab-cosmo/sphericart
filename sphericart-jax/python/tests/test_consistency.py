import jax
import numpy as np
import pytest

import sphericart
import sphericart.jax


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_consistency(xyz, l_max, normalized):
    if normalized:
        calculator = sphericart.SphericalHarmonics(l_max=l_max)
    else:
        calculator = sphericart.SolidHarmonics(l_max=l_max)

    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )
    sph = function(l_max=l_max, xyz=xyz)

    sph_ref = calculator.compute(np.asarray(xyz))

    # some of these values seem to need a larger tolerance
    # even just scaling the input by different numbers (instead of 6)
    # can change the behavior of the test
    np.testing.assert_allclose(sph, sph_ref, rtol=1e-3, atol=1e-8)
