import jax
import sphericart.jax


# Create a random array of Cartesian positions:
key = jax.random.PRNGKey(0)
xyz = 6 * jax.random.normal(key, (10, 3))
l_max = 3  # set l_max to 3
normalized = True  # in this example, we always compute normalized spherical harmonics

# calculate the spherical harmonics with the corresponding function
sph = sphericart.jax.spherical_harmonics(xyz, l_max, normalized)

# jit the function with jax.jit()
jitted_sph_function = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1)

# compute the spherical harmonics with the jitted function and check their values
# against the non-jitted version:
jitted_sph = jitted_sph_function(xyz, l_max, normalized)
assert jax.numpy.allclose(sph, jitted_sph)

# calculate a scalar function 
