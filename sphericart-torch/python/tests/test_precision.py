import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


def test_precision(xyz):
    calculator = sphericart.torch.SolidHarmonics(l_max=4)

    xyz_64 = xyz.clone().to(dtype=torch.float64).detach().requires_grad_(True)
    xyz_32 = xyz.clone().to(dtype=torch.float32).detach().requires_grad_(True)
    assert ((xyz_64.detach() - xyz_32.detach()) ** 2).sum() < 1e-8

    sph_64 = calculator.compute(xyz=xyz_64)
    sph_32 = calculator.compute(xyz=xyz_32)
    assert ((sph_64.detach() / sph_32.detach() - 1) ** 2).sum() < 1e-5

    norm_64 = (sph_64**2).sum()
    norm_32 = (sph_32**2).sum()
    norm_64.backward()
    norm_32.backward()
    assert torch.allclose(xyz_64.grad.detach(), xyz_32.grad.detach().to(torch.float64))
