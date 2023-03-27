import sphericart_torch
import torch


torch.manual_seed(0)


def test_cpu_vs_cuda():
    if torch.cuda.is_available():
        xyz = 6 * torch.randn(
            100, 3, dtype=torch.float64, device="cpu", requires_grad=True
        )
        xyz_cuda = xyz.to("cuda")

        # cartesian spherical harmonics
        calculator = sphericart_torch.SphericalHarmonics(l_max=20, normalized=False)
        sph, _ = calculator.compute(xyz)
        sph_cuda, _ = calculator.compute(xyz_cuda)
        assert torch.allclose(sph, sph_cuda.to("cpu"))

        # normalized spherical harmonics
        calculator = sphericart_torch.SphericalHarmonics(l_max=20, normalized=True)
        sph, _ = calculator.compute(xyz)
        sph_cuda, _ = calculator.compute(xyz_cuda)
        assert torch.allclose(sph, sph_cuda.to("cpu"))
