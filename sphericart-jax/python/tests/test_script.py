import pytest
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import sphericart.jax as sphj


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


def test_script(xyz):
    def compute(xyz):
        sph = sphj.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
        return sph.sum()

    # jit compile the function
    jcompute = jax.jit(compute)
    out = jcompute(xyz)
    # get gradients for the compiled function
    dout = jax.grad(jcompute)(xyz)
