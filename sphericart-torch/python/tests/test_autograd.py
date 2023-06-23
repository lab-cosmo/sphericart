import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    torch.manual_seed(0)
    return 6 * torch.randn(20, 3, dtype=torch.float64, requires_grad=True)


def test_autograd_cartesian(xyz):
    calculator = sphericart.torch.SphericalHarmonics(
        l_max=4, normalized=False, backward_second_derivatives=True
    )

    def compute(xyz):
        sph = calculator.compute(xyz=xyz)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz = xyz.to(device="cuda")
        assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)


def test_autograd_normalized(xyz):
    calculator = sphericart.torch.SphericalHarmonics(
        l_max=4, normalized=True, backward_second_derivatives=True
    )

    def compute(xyz):
        sph = calculator.compute(xyz=xyz)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz = xyz.to(device="cuda")
        assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)


def test_autograd_hessian(xyz):
    # Initialize a calculator with l_max = 1
    calculator = sphericart.torch.SphericalHarmonics(
        l_max=1, normalized=False, backward_second_derivatives=True
    )

    # Fill a single xyz point with arbitrary numbers
    xyz = torch.tensor(
        [
            [0.67, 0.53, -0.22],
        ],
        requires_grad=True,
    )

    # Define a dummy function
    def f(xyz):
        sph = calculator.compute(xyz)[0]  # Discard sample dimension
        return (
            sph[1] ** 2
            + sph[2]
            + 0.20 * sph[1] * sph[3]
            + sph[2] ** 2
            + 0.42 * sph[1] * sph[2]
            + sph[0] ** 12
        )

    hessian = torch.autograd.functional.hessian(f, xyz)[
        0, :, 0, :
    ]  # Discard the two sample dimensions

    # Since sph[0, 1, 2, 3] are proportional to 1, y, z, x respectively,
    # the hessian should be proportional to the following:
    analytical_hessian = torch.tensor(
        [[0.0, 0.2, 0.0], [0.2, 2.0, 0.42], [0.0, 0.42, 2.0]]
    )

    proportionality_factor = analytical_hessian[2, 2] / hessian[2, 2]
    assert torch.allclose(analytical_hessian, hessian * proportionality_factor)
