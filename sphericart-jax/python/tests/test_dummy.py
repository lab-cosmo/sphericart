import numpy as np
import jax
import jax.numpy as jnp
import pytest
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import sphericart.jax
from dummy_sph import dummy_spherical_harmonics


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (20, 3))


def test_jit(xyz):
    jitted_sph = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1,))
    dummy_jitted_sph = jax.jit(dummy_spherical_harmonics, static_argnums=(1,))
    for l_max in [2, 3, 4]:
        sph = jitted_sph(xyz=xyz, l_max=l_max, normalized=True)
        sph_dummy = dummy_jitted_sph(xyz, l_max)
        assert jnp.allclose(sph, sph_dummy)


def test_jacfwd(xyz):
    jacfwd_sph = jax.jacfwd(sphericart.jax.spherical_harmonics)
    dummy_jacfwd_sph = jax.jacfwd(dummy_spherical_harmonics)
    for l_max in [2, 3, 4]:
        sph = jacfwd_sph(xyz, l_max=l_max, normalized=True)
        sph_dummy = dummy_jacfwd_sph(xyz, l_max)
        assert jnp.allclose(sph, sph_dummy)


def test_jacrev(xyz):
    jacrev_sph = jax.jacrev(sphericart.jax.spherical_harmonics)
    dummy_jacrev_sph = jax.jacrev(dummy_spherical_harmonics)
    for l_max in [2, 3, 4]:
        sph = jacrev_sph(xyz, l_max=l_max, normalized=True)
        sph_dummy = dummy_jacrev_sph(xyz, l_max)
        assert jnp.allclose(sph, sph_dummy)


def test_gradgrad(xyz):
    sum_sph = lambda x, l_max, normalized: jnp.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))
    dummy_sum_sph = lambda x, l_max: jnp.sum(dummy_spherical_harmonics(x, l_max))
    sum_grad_sph = lambda x, l_max, normalized: jnp.sum(jax.grad(sum_sph)(x, l_max, normalized))
    dummy_sum_grad_sph = lambda x, l_max: jnp.sum(jax.grad(dummy_sum_sph)(x, l_max))
    gradgrad_sph = jax.grad(sum_grad_sph)
    dummy_gradgrad_sph = jax.grad(dummy_sum_grad_sph)
    for l_max in [2, 3, 4]:
        sph = gradgrad_sph(xyz, l_max, normalized=True)
        sph_dummy = dummy_gradgrad_sph(xyz, l_max)
        assert jnp.allclose(sph, sph_dummy, atol=1e-10)
