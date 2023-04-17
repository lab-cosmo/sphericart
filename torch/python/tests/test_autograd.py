import pytest

import sphericart_torch
import torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


def test_autograd_cartesian(xyz):
    calculator = sphericart_torch.SphericalHarmonics(l_max=4, normalized=False)

    def compute(xyz):
        sph, _ = calculator.compute(xyz=xyz)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz = xyz.to(device="cuda")
        assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)


def test_autograd_normalized(xyz):
    calculator = sphericart_torch.SphericalHarmonics(l_max=4, normalized=True)

    def compute(xyz):
        sph, _ = calculator.compute(xyz=xyz)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz = xyz.to(device="cuda")
        assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)
