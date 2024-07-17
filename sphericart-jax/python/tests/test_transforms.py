import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sphericart.jax


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (10, 3))


def test_script(xyz):
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
        return sph.sum()

    # jit compile the function
    jcompute = jax.jit(compute)
    jcompute(xyz)
    # get gradients for the compiled function
    jax.grad(jcompute)(xyz)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_jit(xyz, l_max, normalized):
    jitted_sph = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1, 2))
    calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
    sph = jitted_sph(xyz=xyz, l_max=l_max, normalized=normalized)
    sph_ref = calculator.compute(np.asarray(xyz))
    np.testing.assert_allclose(sph, sph_ref, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap(xyz, l_max, normalized):
    vmapped_sph = jax.vmap(sphericart.jax.spherical_harmonics, in_axes=(0, None, None))
    calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
    sph = vmapped_sph(xyz, l_max, normalized)
    sph_ref = calculator.compute(np.asarray(xyz))
    np.testing.assert_allclose(sph, sph_ref, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_jit_jacfwd(xyz, l_max, normalized):
    transformed_sph = jax.jit(
        jax.jacfwd(sphericart.jax.spherical_harmonics), static_argnums=(1, 2)
    )
    transformed_sph(xyz, l_max, normalized)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_hessian_jit(xyz, l_max, normalized):
    transformed_sph = jax.hessian(
        jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1, 2))
    )
    transformed_sph(xyz, l_max, normalized)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap_grad(xyz, l_max, normalized):
    def single_scalar_output(x, l_max, normalized):
        return jnp.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))

    single_grad = jax.grad(single_scalar_output)
    sh_grad = jax.vmap(single_grad, in_axes=(0, None, None), out_axes=0)
    sh_grad(xyz, l_max, normalized)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap_hessian(xyz, l_max, normalized):
    def single_scalar_output(x, l_max, normalized):
        return jnp.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))

    single_hessian = jax.hessian(single_scalar_output)
    sh_hessian = jax.vmap(single_hessian, in_axes=(0, None, None), out_axes=0)
    sh_hessian(xyz, l_max, normalized)
