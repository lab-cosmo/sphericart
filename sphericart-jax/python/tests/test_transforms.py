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
        sph = sphericart.jax.solid_harmonics(l_max=4, xyz=xyz)
        return sph.sum()

    # jit compile the function
    jcompute = jax.jit(compute)
    jcompute(xyz)
    # get gradients for the compiled function
    jax.grad(jcompute)(xyz)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_jit(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )
    jitted_sph = jax.jit(function, static_argnums=(1,))
    if normalized:
        calculator = sphericart.SphericalHarmonics(l_max=l_max)
    else:
        calculator = sphericart.SolidHarmonics(l_max=l_max)
    sph = jitted_sph(xyz=xyz, l_max=l_max)
    sph_ref = calculator.compute(np.asarray(xyz))
    np.testing.assert_allclose(sph, sph_ref, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )
    vmapped_sph = jax.vmap(function, in_axes=(0, None))
    if normalized:
        calculator = sphericart.SphericalHarmonics(l_max=l_max)
    else:
        calculator = sphericart.SolidHarmonics(l_max=l_max)
    sph = vmapped_sph(xyz, l_max)
    sph_ref = calculator.compute(np.asarray(xyz))
    np.testing.assert_allclose(sph, sph_ref, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_jit_jacfwd(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )
    transformed_sph = jax.jit(jax.jacfwd(function), static_argnums=(1,))
    transformed_sph(xyz, l_max)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_hessian_jit(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )
    transformed_sph = jax.hessian(jax.jit(function, static_argnums=(1,)))
    transformed_sph(xyz, l_max)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap_grad(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )

    def single_scalar_output(x, l_max):
        return jnp.sum(function(x, l_max))

    single_grad = jax.grad(single_scalar_output)
    sh_grad = jax.vmap(single_grad, in_axes=(0, None), out_axes=0)
    sh_grad(xyz, l_max)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("l_max", [4, 7, 10])
def test_vmap_hessian(xyz, l_max, normalized):
    function = (
        sphericart.jax.spherical_harmonics
        if normalized
        else sphericart.jax.solid_harmonics
    )

    def single_scalar_output(x, l_max):
        return jnp.sum(function(x, l_max))

    single_hessian = jax.hessian(single_scalar_output)
    sh_hessian = jax.vmap(single_hessian, in_axes=(0, None), out_axes=0)
    sh_hessian(xyz, l_max)
