import equinox as eqx
import jax
import jax.numpy as jnp

import sphericart.jax


def test_nn():
    class NN(eqx.Module):
        def __init__(self):
            pass

        def __call__(self, xyz):
            sph = sphericart.jax.spherical_harmonics(xyz, 4, True)
            sum = jnp.sum(sph)
            return sum

    random_key = jax.random.PRNGKey(123)
    xyz = jax.random.normal(random_key, (10, 3))
    nn = NN()
    value_grad_nn = jax.jit(jax.value_and_grad(nn))
    value, grad = value_grad_nn(xyz)
