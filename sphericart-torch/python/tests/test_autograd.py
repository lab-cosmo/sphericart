import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    torch.manual_seed(0)
    return 6 * torch.randn(20, 3, dtype=torch.float64, requires_grad=True)


def test_autograd_cartesian(xyz):
    calculator = sphericart.torch.SolidHarmonics(
        l_max=4, backward_second_derivatives=True
    )

    def compute(xyz_in):
        sph = calculator.compute(xyz=xyz_in)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().cuda().requires_grad_(True)
        assert torch.autograd.gradcheck(compute, xyz_cuda, fast_mode=True)
        assert torch.autograd.gradgradcheck(compute, xyz_cuda, fast_mode=True)


def test_autograd_normalized(xyz):
    calculator = sphericart.torch.SolidHarmonics(
        l_max=4, backward_second_derivatives=True
    )

    def compute(xyz_in):
        sph = calculator.compute(xyz=xyz_in)
        assert torch.linalg.norm(sph) != 0.0
        return sph

    assert torch.autograd.gradcheck(compute, xyz, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, xyz, fast_mode=True)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().cuda().requires_grad_(True)
        assert torch.autograd.gradcheck(compute, xyz_cuda, fast_mode=True)
        assert torch.autograd.gradgradcheck(compute, xyz_cuda, fast_mode=True)


def test_autograd_hessian(xyz):
    # Initialize a calculator with l_max = 1
    calculator = sphericart.torch.SolidHarmonics(
        l_max=1, backward_second_derivatives=True
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


def test_second_derivative_error(xyz):
    # This is a problematic test, as it tests some potential silent failures

    # Initialize a calculator
    calculator = sphericart.torch.SphericalHarmonics(
        l_max=8, backward_second_derivatives=False
    )

    # Fill a single xyz point with arbitrary numbers
    xyz = torch.tensor(
        [
            [0.67, 0.53, -0.22],
        ],
        requires_grad=True,
    )

    # Compute the spherical harmonics and run backward 2 times.
    # The second one is supposed to raise an error.
    sph = calculator.compute(xyz)
    l0 = torch.sum(sph)
    d1 = torch.autograd.grad(
        outputs=l0,
        inputs=xyz,
        retain_graph=True,
        create_graph=True,
    )[0]
    l1 = torch.sum(d1)

    # case 1: autograd.grad raises an error
    # being the first time the second derivatives are requested, and since
    # `backward_second_derivatives=False`, a warning is also displayed
    with pytest.warns(
        UserWarning,
        match="Second derivatives of the spherical harmonics with respect "
        "to the Cartesian coordinates were not requested at class creation.",
    ):
        with pytest.raises(
            RuntimeError,
            match="One of the differentiated Tensors appears to not have "
            "been used in the graph. Set allow_unused=True if this is the "
            "desired behavior.",
        ):
            torch.autograd.grad(
                outputs=l1,
                inputs=xyz,
                retain_graph=True,
                create_graph=True,
            )

    # case 2: autograd.grad with allow_unused=True fails silently
    torch.autograd.grad(
        outputs=l1,
        inputs=xyz,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )

    # case 3: backward fails silently
    l1.backward()

    # case 4: autograd.functional.hessian fails silently
    def f(xyz):  # dummy function
        sph = calculator.compute(xyz)[0]  # Discard sample dimension
        return torch.sum(sph)

    torch.autograd.functional.hessian(f, xyz)[
        0, :, 0, :
    ]  # Discard the two sample dimensions


def test_third_derivative_error(xyz):
    # Initialize a calculator
    calculator = sphericart.torch.SphericalHarmonics(
        l_max=8, backward_second_derivatives=True
    )

    # Fill a single xyz point with arbitrary numbers
    xyz = torch.tensor(
        [
            [0.67, 0.53, -0.22],
        ],
        requires_grad=True,
    )

    # Compute the spherical harmonics and run backward 3 times.
    # The third one must raise.
    sph = calculator.compute(xyz)
    l0 = torch.sum(sph)
    d1 = torch.autograd.grad(
        outputs=l0,
        inputs=xyz,
        retain_graph=True,
        create_graph=True,
    )[0]
    l1 = torch.sum(d1)
    d2 = torch.autograd.grad(
        outputs=l1,
        inputs=xyz,
        retain_graph=True,
        create_graph=True,
    )[0]
    s2 = torch.sum(d2)
    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        torch.autograd.grad(
            outputs=s2,
            inputs=xyz,
            retain_graph=False,
            create_graph=False,
        )
    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        s2.backward()
