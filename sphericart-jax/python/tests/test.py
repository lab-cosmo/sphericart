import sphericart
import sphericart.jax
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)
xyz = 6 * jax.random.normal(key, (100, 3))

for l_max in [4, 7, 10]:
    for normalized in [True, False]:
        calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
        sph = sphericart.jax.spherical_harmonics(xyz, l_max, normalized)
        sph_ref = calculator.compute(np.asarray(xyz))
        np.testing.assert_allclose(sph, sph_ref)

jitted_sph = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1,))
for l_max in [4, 7, 10]:
    for normalized in [True, False]:
        calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
        sph = jitted_sph(xyz=xyz, l_max=l_max, normalized=normalized)
        sph_ref = calculator.compute(np.asarray(xyz))
        np.testing.assert_allclose(sph, sph_ref)

vmapped_sph = jax.vmap(sphericart.jax.spherical_harmonics, in_axes=(0, None, None))
for l_max in [4, 7, 10]:
    for normalized in [True, False]:
        calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
        sph = vmapped_sph(xyz, l_max, normalized)
        sph_ref = calculator.compute(np.asarray(xyz))
        np.testing.assert_allclose(sph, sph_ref)

from sphericart.jax import dsph
for l_max in [4, 7, 10]:
    for normalized in [True, False]:
        calculator = sphericart.SphericalHarmonics(l_max=l_max, normalized=normalized)
        sph, dsph_actual = dsph(xyz, l_max, normalized)
        sph, dsph_ref = calculator.compute_with_gradients(np.asarray(xyz))
        np.testing.assert_allclose(dsph_actual, dsph_ref)
