import jax
import jax.numpy as jnp
import jax.test_util as jtu
import pytest

import sphericart.jax


@pytest.mark.parametrize("normalized", [True, False], ids=["spherical", "solid"])
def test_autograd(normalized):
    # here, 32-bit numerical gradients are very noisy, so we use 64-bit
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)
    xyz = 6 * jax.random.normal(key, (100, 3))

    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )

    def compute(xyz):
        sph = function(xyz, 4)
        return jnp.sum(sph)

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=1)

    # reset to 32-bit so this doesn't carry over to other tests
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("normalized", [True, False], ids=["spherical", "solid"])
def test_autograd_second_derivatives(normalized):
    # here, 32-bit numerical gradients are very noisy, so we use 64-bit
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)
    xyz = 6 * jax.random.normal(key, (100, 3))

    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )

    def compute(xyz):
        sph = function(xyz, 4)
        return jnp.sum(sph)

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=2)

    # reset to 32-bit so this doesn't carry over to other tests
    jax.config.update("jax_enable_x64", False)
