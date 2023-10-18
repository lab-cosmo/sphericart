import jax
import jax.numpy as jnp
import sphericart.jax


#

key = jax.random.PRNGKey(0)
xyz = 6 * jax.random.normal(key, (10, 3))

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


def test_hessian_jit(xyz):
    transformed_sph = jax.hessian(jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1))
    for l_max in [4, 7, 10]:
        for normalized in [True, False]:
            sph = transformed_sph(xyz, l_max, normalized)
