import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    torch.manual_seed(0)
    return 6 * torch.randn(20, 3, dtype=torch.float64, requires_grad=True)


def test_autograd_cartesian(xyz):
    calculator = sphericart.torch.SolidHarmonics(l_max=4)

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
    calculator = sphericart.torch.SolidHarmonics(l_max=4)

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
    calculator = sphericart.torch.SolidHarmonics(l_max=1)

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


@pytest.mark.parametrize("normalized", [True, False])
def test_noncontiguous_input(normalized):
    calculator_cls = (
        sphericart.torch.SphericalHarmonics
        if normalized
        else sphericart.torch.SolidHarmonics
    )
    calculator = calculator_cls(l_max=4)

    def check(device):
        base = torch.randn(20, 6, dtype=torch.float64, device=device)
        xyz = base[:, ::2].detach().requires_grad_(True)
        xyz_ref = xyz.detach().clone().contiguous().requires_grad_(True)

        value = calculator.compute(xyz)
        value_ref = calculator.compute(xyz_ref)
        assert torch.allclose(value, value_ref)

        value.sum().backward()
        value_ref.sum().backward()
        assert torch.allclose(xyz.grad, xyz_ref.grad)

        grad = calculator.compute_with_gradients(xyz)
        grad_ref = calculator.compute_with_gradients(xyz_ref)
        assert torch.allclose(grad[0], grad_ref[0])
        assert torch.allclose(grad[1], grad_ref[1])

        hessian = calculator.compute_with_hessians(xyz)
        hessian_ref = calculator.compute_with_hessians(xyz_ref)
        assert torch.allclose(hessian[0], hessian_ref[0])
        assert torch.allclose(hessian[1], hessian_ref[1])
        assert torch.allclose(hessian[2], hessian_ref[2])

    check("cpu")
    if torch.cuda.is_available():
        check("cuda")


@pytest.mark.parametrize("normalized", [True, False])
def test_compute_with_gradients_output_is_differentiable(normalized):
    calculator_cls = (
        sphericart.torch.SphericalHarmonics
        if normalized
        else sphericart.torch.SolidHarmonics
    )
    calculator = calculator_cls(l_max=4)

    xyz = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    _, dsph = calculator.compute_with_gradients(xyz)
    grad = torch.autograd.grad(dsph.sum(), xyz, create_graph=True)[0]
    assert grad.requires_grad
    assert grad.shape == xyz.shape

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().cuda().requires_grad_(True)
        _, dsph_cuda = calculator.compute_with_gradients(xyz_cuda)
        grad_cuda = torch.autograd.grad(dsph_cuda.sum(), xyz_cuda, create_graph=True)[0]
        assert grad_cuda.requires_grad
        assert grad_cuda.shape == xyz_cuda.shape


@pytest.mark.parametrize("normalized", [True, False])
def test_compute_with_hessians_stops_at_third_derivatives(normalized):
    calculator_cls = (
        sphericart.torch.SphericalHarmonics
        if normalized
        else sphericart.torch.SolidHarmonics
    )
    calculator = calculator_cls(l_max=4)

    xyz = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    _, _, ddsph = calculator.compute_with_hessians(xyz)
    with pytest.raises(
        RuntimeError,
        match=(
            "Third derivatives of the spherical harmonics with respect to the "
            "Cartesian coordinates are not supported."
        ),
    ):
        torch.autograd.grad(ddsph.sum(), xyz)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().cuda().requires_grad_(True)
        _, _, ddsph_cuda = calculator.compute_with_hessians(xyz_cuda)
        with pytest.raises(
            RuntimeError,
            match=(
                "Third derivatives of the spherical harmonics with respect to the "
                "Cartesian coordinates are not supported."
            ),
        ):
            torch.autograd.grad(ddsph_cuda.sum(), xyz_cuda)


def test_second_derivative_supported_by_default(xyz):
    calculator = sphericart.torch.SphericalHarmonics(l_max=8)

    # Fill a single xyz point with arbitrary numbers
    xyz = torch.tensor(
        [
            [0.67, 0.53, -0.22],
        ],
        requires_grad=True,
    )

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
    assert d2.shape == xyz.shape


def test_third_derivative_error(xyz):
    calculator = sphericart.torch.SphericalHarmonics(l_max=8)

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
        match=(
            "Third derivatives of the spherical harmonics with respect to the "
            "Cartesian coordinates are not supported."
        ),
    ):
        torch.autograd.grad(
            outputs=s2,
            inputs=xyz,
            retain_graph=False,
            create_graph=False,
        )
    with pytest.raises(
        RuntimeError,
        match="Trying to backward through the graph a second time",
    ):
        s2.backward()
