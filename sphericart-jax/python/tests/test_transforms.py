import numpy as np
import pytest
import jax
jax.config.update("jax_platform_name", "cpu")
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
    out = jcompute(xyz)
    # get gradients for the compiled function
    dout = jax.grad(jcompute)(xyz)


def test_jit(xyz):
    jitted_sph = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1,))
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
            sph = jitted_sph(xyz=xyz, l_max=l_max, normalized=normalized)
            sph_ref = calculator.compute(np.asarray(xyz))
            np.testing.assert_allclose(sph, sph_ref)

            
def test_vmap(xyz):
    vmapped_sph = jax.vmap(sphericart.jax.spherical_harmonics, in_axes=(0, None, None))
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
            sph = vmapped_sph(xyz, l_max, normalized)
            sph_ref = calculator.compute(np.asarray(xyz))
            np.testing.assert_allclose(sph, sph_ref)


def test_jit_jacfwd(xyz):
    transformed_sph = jax.jit(jax.jacfwd(sphericart.jax.spherical_harmonics), static_argnums=1)
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            sph = transformed_sph(xyz, l_max, normalized)


def test_vmap_grad(xyz):
    def single_scalar_output(x, l_max, normalized):        
        return jax.numpy.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))
    single_grad = jax.grad(single_scalar_output)
    sh_grad = jax.vmap(single_grad, in_axes=(0, None, None), out_axes=0)
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            _ = sh_grad(xyz, l_max, normalized)


def test_vmap_hessian(xyz):
    def single_scalar_output(x, l_max, normalized):        
        return jax.numpy.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))
    single_hessian = jax.hessian(single_scalar_output)
    sh_hessian = jax.vmap(single_hessian, in_axes=(0, None, None), out_axes=0)
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            _ = sh_hessian(xyz, l_max, normalized)
