import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return 6 * torch.randn(10, 3, dtype=torch.float64, requires_grad=True)


def test_cpu_vs_cuda(xyz):
    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")

        calculator = sphericart.torch.SolidHarmonics(l_max=20)
        sph, grad_sph = calculator.compute_with_gradients(xyz)
        sph_cuda, grad_sph_cuda = calculator.compute_with_gradients(xyz_cuda)
        print((sph - sph_cuda.to("cpu")).flatten().max())
        print((sph - sph_cuda.to("cpu"))[0])
        print((sph - sph_cuda.to("cpu"))[1])
        assert torch.allclose(sph, sph_cuda.to("cpu"), rtol=1e-7)
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"), rtol=1e-7)

        calculator = sphericart.torch.SphericalHarmonics(l_max=20)
        sph, grad_sph = calculator.compute_with_gradients(xyz)
        sph_cuda, grad_sph_cuda = calculator.compute_with_gradients(xyz_cuda)

        print ("--grads--")
        print (grad_sph)
        print (grad_sph_cuda)
        
        assert torch.allclose(sph, sph_cuda.to("cpu"), rtol=1e-7)
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"), rtol=1e-7)
