import pytest
import jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax._src.test_util as jtu
import sphericart.jax


@pytest.fixture
def xyz():
    key = jax.random.PRNGKey(0)
    return 6 * jax.random.normal(key, (100, 3))


@pytest.mark.parametrize("normalized", [True, False])
def test_autograd(xyz, normalized):
    # print(xyz.device_buffer.device())
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(xyz=xyz, l_max=4, normalized=normalized)
        assert jnp.linalg.norm(sph) != 0.0
        return sph.sum()

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=1)


@pytest.mark.parametrize("normalized", [True, False])
def test_autograd_second_derivatives(xyz, normalized):
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(xyz=xyz, l_max=4, normalized=normalized)
        assert jnp.linalg.norm(sph) != 0.0
        return sph.sum()

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=2)
