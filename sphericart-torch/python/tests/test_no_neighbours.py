import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.fixture
def xyz():
    return torch.zeros(0, 3, dtype=torch.float64, requires_grad=True)


def test_error(xyz):

    calculator = sphericart.torch.SphericalHarmonics(l_max=20, normalized=False)

    try:
        sph = calculator.compute(xyz)
    except RuntimeError as e:
        pytest.fail(f"compute threw an error: {e}")

    try:
        sph, grad_sph = calculator.compute_with_gradients(xyz)
    except RuntimeError as e:
        pytest.fail(f"compute_with_gradients threw an error: {e}")

    try:
        sph, grad_sph, hess_sph = calculator.compute_with_hessians(xyz)
    except RuntimeError as e:
        pytest.fail(f"compute_with_hessians threw an error: {e}")

    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")
        try:
            sph = calculator.compute(xyz_cuda)
        except RuntimeError as e:
            pytest.fail(f"compute threw an error: {e}")

        try:
            sph, grad_sph = calculator.compute_with_gradients(xyz_cuda)
        except RuntimeError as e:
            pytest.fail(f"compute_with_gradients threw an error: {e}")

        try:
            sph, grad_sph, hess_sph = calculator.compute_with_hessians(xyz_cuda)
        except RuntimeError as e:
            pytest.fail(f"compute_with_hessians threw an error: {e}")
