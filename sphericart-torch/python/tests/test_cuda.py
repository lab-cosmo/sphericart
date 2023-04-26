import pytest

import sphericart.torch
import torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


def test_cpu_vs_cuda(xyz):
    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")

        calculator = sphericart.torch.SphericalHarmonics(l_max=20, normalized=False)
        sph, grad_sph = calculator.compute_with_gradients(xyz)
        sph_cuda, grad_sph_cuda = calculator.compute_with_gradients(xyz_cuda)

        assert torch.allclose(sph, sph_cuda.to("cpu"))
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"))

        calculator = sphericart.torch.SphericalHarmonics(l_max=20, normalized=True)
        sph, grad_sph = calculator.compute_with_gradients(xyz)
        sph_cuda, grad_sph_cuda = calculator.compute_with_gradients(xyz_cuda)

        assert torch.allclose(sph, sph_cuda.to("cpu"))
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"))
