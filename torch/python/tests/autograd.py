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
        return sphericart_torch.SphericalHarmonics(l_max=20, normalized=False).compute(
            xyz=xyz
        )

    return _compute_unnormalized


@pytest.fixture
def compute_normalized(xyz):
    def _compute_normalized(xyz):
        return sphericart_torch.SphericalHarmonics(l_max=20, normalized=True).compute(
            xyz=xyz
        )

    return _compute_normalized


def test_autograd_cartesian(compute_unnormalized, xyz):
    assert torch.autograd.gradcheck(compute_unnormalized, xyz, fast_mode=True)


def test_autograd_normalized(compute_normalized, xyz):
    assert torch.autograd.gradcheck(compute_normalized, xyz, fast_mode=True)


if __name__ == "__main__":
    pytest.main([__file__])
