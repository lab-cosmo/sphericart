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


def test_autograd_cartesian(xyz):
    # print(xyz.device_buffer.device())
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
        assert jnp.linalg.norm(sph) != 0.0
        return sph.sum()

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=1)


def test_autograd_normalized(xyz):
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=True, xyz=xyz)
        assert jnp.linalg.norm(sph) != 0.0
        return sph

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=1)


def test_autograd_cartesian_second_derivatives(xyz):
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=False, xyz=xyz)
        assert jnp.linalg.norm(sph) != 0.0
        return sph.sum()

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=2)


def test_autograd_normalized_second_derivatives(xyz):
    def compute(xyz):
        sph = sphericart.jax.spherical_harmonics(l_max=4, normalized=True, xyz=xyz)
        assert jnp.linalg.norm(sph) != 0.0
        return sph

    jtu.check_grads(compute, (xyz,), modes=["fwd", "bwd"], order=2)
