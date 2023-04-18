import sphericart_torch
import torch


torch.manual_seed(0)


def test_cpu_vs_cuda():
    if torch.cuda.is_available():
        xyz = 6 * torch.randn(
            100, 3, dtype=torch.float64, device="cpu", requires_grad=True
        )
        xyz_cuda = xyz.to("cuda")

        calculator = sphericart_torch.SphericalHarmonics(l_max=20, normalized=False)
        sph, grad_sph = calculator.compute(xyz, gradients=True)
        sph_cuda, grad_sph_cuda = calculator.compute(xyz_cuda, gradients=True)

        assert torch.allclose(sph, sph_cuda.to("cpu"))
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"))

        calculator = sphericart_torch.SphericalHarmonics(l_max=20, normalized=True)
        sph, grad_sph = calculator.compute(xyz, gradients=True)
        sph_cuda, grad_sph_cuda = calculator.compute(xyz_cuda, gradients=True)

        assert torch.allclose(sph, sph_cuda.to("cpu"))
        assert torch.allclose(grad_sph, grad_sph_cuda.to("cpu"))


test_cpu_vs_cuda()
