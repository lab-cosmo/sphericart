import pytest
import sphericart_torch

import torch

torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


@pytest.fixture
def compute_unnormalized(xyz):
    def _compute_unnormalized(xyz):
        return sphericart_torch.SphericalHarmonics(l_max=4, normalized=False).compute(
            xyz=xyz
        )[0]

    return _compute_unnormalized


@pytest.fixture
def compute_normalized(xyz):
    def _compute_normalized(xyz):
        return sphericart_torch.SphericalHarmonics(l_max=4, normalized=True).compute(
            xyz=xyz
        )[0]

    return _compute_normalized


def test_autograd_cartesian(compute_unnormalized, xyz):
    assert torch.autograd.gradcheck(compute_unnormalized, xyz, fast_mode=True)


def test_autograd_normalized(compute_normalized, xyz):
    assert torch.autograd.gradcheck(compute_normalized, xyz, fast_mode=True)


def test_precision_base(compute_normalized, xyz):
    d_xyz = xyz.clone().detach().requires_grad_(True)
    d_sph = sphericart_torch.SphericalHarmonics(l_max=10, normalized=False).compute(
        d_xyz
    )[0]
    d_norm = (d_sph**2).sum()
    d_norm.backward()


def test_precision(compute_normalized, xyz):
    d_xyz = xyz.clone().to(dtype=torch.float32).detach().requires_grad_(True)
    s_xyz = xyz.clone().to(dtype=torch.float32).detach().requires_grad_(True)
    assert ((d_xyz.detach() - s_xyz.detach()) ** 2).sum() < 1e-8

    d_sph = sphericart_torch.SphericalHarmonics(l_max=10, normalized=False).compute(
        xyz=d_xyz
    )[0]
    s_sph = sphericart_torch.SphericalHarmonics(l_max=10, normalized=False).compute(
        xyz=s_xyz
    )[0]
    assert ((d_sph.detach() / s_sph.detach() - 1) ** 2).sum() < 1e-5

    d_norm = (d_sph**2).sum()
    s_norm = (s_sph**2).sum()
    d_norm.backward()
    s_norm.backward()
    assert torch.allclose(d_xyz.grad.detach(), s_xyz.grad.detach())


if __name__ == "__main__":
    pytest.main([__file__])
