import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


def test_cpu_vs_cuda(xyz):
    if torch.cuda.is_available():
        calculator = sphericart.torch.SphericalHarmonics(l_max=20, normalized=False)

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        with torch.cuda.stream(s1):
            sph_1, grad_sph_1 = calculator.compute_with_gradients(xyz)

        with torch.cuda.stream(s2):
            sph_2, grad_sph_2 = calculator.compute_with_gradients(xyz)

        assert torch.allclose(sph_1, sph_2)
        assert torch.allclose(grad_sph_1, grad_sph_2)
