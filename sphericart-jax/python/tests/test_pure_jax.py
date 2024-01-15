import numpy as np
import jax
import jax.numpy as jnp
import pytest

# jax.config.update("jax_platform_name", "cpu")
import sphericart.jax
from pure_jax_sph import pure_jax_spherical_harmonics


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (20, 3))


@pytest.mark.parametrize("l_max", [2, 7])
def test_jit(xyz, l_max):
    jitted_sph = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1,))
    pure_jax_jitted_sph = jax.jit(pure_jax_spherical_harmonics, static_argnums=(1,))
    sph = jitted_sph(xyz=xyz, l_max=l_max, normalized=True)
    sph_pure_jax = pure_jax_jitted_sph(xyz, l_max)
    assert jnp.allclose(sph, sph_pure_jax)


@pytest.mark.parametrize("l_max", [2, 7])
def test_jacfwd(xyz, l_max):
    jacfwd_sph = jax.jacfwd(sphericart.jax.spherical_harmonics)
    pure_jax_jacfwd_sph = jax.jacfwd(pure_jax_spherical_harmonics)
    sph = jacfwd_sph(xyz, l_max=l_max, normalized=True)
    sph_pure_jax = pure_jax_jacfwd_sph(xyz, l_max)
    assert jnp.allclose(sph, sph_pure_jax)


@pytest.mark.parametrize("l_max", [2, 7])
def test_jacrev(xyz, l_max):
    jacrev_sph = jax.jacrev(sphericart.jax.spherical_harmonics)
    pure_jax_jacrev_sph = jax.jacrev(pure_jax_spherical_harmonics)
    sph = jacrev_sph(xyz, l_max=l_max, normalized=True)
    sph_pure_jax = pure_jax_jacrev_sph(xyz, l_max)
    assert jnp.allclose(sph, sph_pure_jax)


@pytest.mark.parametrize("l_max", [2, 7])
def test_gradgrad(xyz, l_max):
    sum_sph = lambda x, l_max, normalized: jnp.sum(
        sphericart.jax.spherical_harmonics(x, l_max, normalized)
    )
    pure_jax_sum_sph = lambda x, l_max: jnp.sum(pure_jax_spherical_harmonics(x, l_max))
    sum_grad_sph = lambda x, l_max, normalized: jnp.sum(
        jax.grad(sum_sph)(x, l_max, normalized)
    )
    pure_jax_sum_grad_sph = lambda x, l_max: jnp.sum(
        jax.grad(pure_jax_sum_sph)(x, l_max)
    )
    gradgrad_sph = jax.grad(sum_grad_sph)
    pure_jax_gradgrad_sph = jax.grad(pure_jax_sum_grad_sph)
    sph = gradgrad_sph(xyz, l_max, normalized=True)
    sph_pure_jax = pure_jax_gradgrad_sph(xyz, l_max)
    assert jnp.allclose(sph, sph_pure_jax)
