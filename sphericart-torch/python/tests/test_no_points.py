import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("l_max", [0, 3, 7, 10, 20, 50])
def test_error(l_max, normalized):
    xyz = torch.zeros(0, 3, dtype=torch.float64)

    calculator = sphericart.torch.SphericalHarmonics(l_max=l_max, normalized=normalized)

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
