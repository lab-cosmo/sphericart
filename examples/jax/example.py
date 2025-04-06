import jax
import sphericart.jax


# Create a random array of Cartesian positions:
key = jax.random.PRNGKey(0)
xyz = 6 * jax.random.normal(key, (10, 3))
l_max = 3  # set l_max to 3

# calculate the spherical harmonics with the corresponding function,
# we could also compute the solid harmonics with sphericart.jax.solid_harmonics
sph = sphericart.jax.spherical_harmonics(xyz, l_max)

# jit the function with jax.jit()
# the l_max argument (position 1 in the signature) must be static
jitted_sph_function = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1,))

# compute the spherical harmonics with the jitted function and check their values
# against the non-jitted version
jitted_sph = jitted_sph_function(xyz, l_max)
assert jax.numpy.allclose(sph, jitted_sph)


# calculate a scalar function of the spherical harmonics and take its gradient
# with respect to the input Cartesian coordinates, as well as its hessian
def scalar_output(xyz, l_max):
    return jax.numpy.sum(sphericart.jax.spherical_harmonics(xyz, l_max))

grad = jax.grad(scalar_output)(xyz, l_max)

# NB: this computes a (n_samples,3,n_samples,3) hessian, i.e. includes cross terms
# between samples.
hessian = jax.hessian(scalar_output)(xyz, l_max)

# usually you want a hessian in the shape (n_samples, 3, 3), taking derivatives
# wrt the coordinates of the same sample. one way to achieve this is as follows

def single_scalar_output(xyz, l_max):
    return jax.numpy.sum(sphericart.jax.spherical_harmonics(xyz, l_max))

# define a function that computes the Hessian for a single (3,) input
single_hessian = jax.hessian(single_scalar_output)

# use vmap to vectorize the Hessian computation over the first axis
sh_hess = jax.vmap(single_hessian, in_axes=(0, None))


# calculate a function of the spherical harmonics that returns an array
# and take its jacobian with respect to the input Cartesian coordinates,
# both in forward mode and in reverse mode
def array_output(xyz, l_max):
    return jax.numpy.sum(
        sphericart.jax.spherical_harmonics(xyz, l_max), axis=0
    )

jacfwd = jax.jacfwd(array_output)(xyz, l_max)
jacrev = jax.jacrev(array_output)(xyz, l_max)

# use vmap and compare the result with the original result:
vmapped_sph = jax.vmap(sphericart.jax.spherical_harmonics, in_axes=(0, None))(
    xyz, l_max
)
assert jax.numpy.allclose(sph, vmapped_sph)
