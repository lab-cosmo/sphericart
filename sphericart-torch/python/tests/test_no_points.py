import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("l_max", [0, 3, 7, 10, 20, 50])
def test_error(l_max, normalized):
    xyz = torch.zeros(0, 3, dtype=torch.float64)

    if normalized:
        calculator = sphericart.torch.SphericalHarmonics(l_max=l_max)
    else:
        calculator = sphericart.torch.SolidHarmonics(l_max=l_max)

    sph = calculator.compute(xyz)
    assert sph.shape == (0, (l_max + 1) ** 2)

    sph, grad_sph = calculator.compute_with_gradients(xyz)
    assert sph.shape == (0, (l_max + 1) ** 2)
    assert grad_sph.shape == (0, 3, (l_max + 1) ** 2)

    sph, grad_sph, hess_sph = calculator.compute_with_hessians(xyz)
    assert sph.shape == (0, (l_max + 1) ** 2)
    assert grad_sph.shape == (0, 3, (l_max + 1) ** 2)
    assert hess_sph.shape == (0, 3, 3, (l_max + 1) ** 2)

    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")

        sph = calculator.compute(xyz_cuda)
        assert sph.shape == (0, (l_max + 1) ** 2)

        sph, grad_sph = calculator.compute_with_gradients(xyz_cuda)
        assert sph.shape == (0, (l_max + 1) ** 2)
        assert grad_sph.shape == (0, 3, (l_max + 1) ** 2)

        sph, grad_sph, hess_sph = calculator.compute_with_hessians(xyz_cuda)
        assert sph.shape == (0, (l_max + 1) ** 2)
        assert grad_sph.shape == (0, 3, (l_max + 1) ** 2)
        assert hess_sph.shape == (0, 3, 3, (l_max + 1) ** 2)
