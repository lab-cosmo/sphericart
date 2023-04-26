import itertools

import pytest

import sphericart.torch
import torch


# only run e3nn tests if e3nn is present
try:
    import e3nn

    _HAS_E3NN = True
except ModuleNotFoundError:
    _HAS_E3NN = False

torch.manual_seed(0)
TOLERANCE = 1e-10


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


# only include tests if e3nn is available
if _HAS_E3NN:

    def relative_mse(a, b):
        return ((a - b) ** 2).sum() / ((a + b) ** 2).sum()

    def test_e3nn_inputs(xyz):
        """Checks that the wrapper accepts the arguments it can accept."""

        # clone so we can backward without messing up xyz
        xyz_sh = xyz.clone().detach().requires_grad_()
        xyz_e3nn = xyz.clone().detach().requires_grad_()

        e3nn_reference = e3nn.o3.spherical_harmonics(8, xyz, False)
        sh = sphericart.torch.e3nn_spherical_harmonics(8, xyz, False)

        assert relative_mse(e3nn_reference.detach(), sh.detach()) < TOLERANCE

        e3nn_reference = e3nn.o3.spherical_harmonics([1, 3, 5], xyz_e3nn, True)
        sh = sphericart.torch.e3nn_spherical_harmonics([1, 3, 5], xyz_sh, True)

        assert relative_mse(e3nn_reference.detach(), sh.detach()) < TOLERANCE

        # now test autograd
        e3nn_sum = torch.sum(e3nn_reference)
        sh_sum = torch.sum(sh)

        e3nn_sum.backward()
        sh_sum.backward()

        assert relative_mse(xyz_e3nn.grad.detach(), xyz_sh.grad.detach()) < TOLERANCE

    @pytest.mark.parametrize(
        "normalize, normalization",
        list(itertools.product([True, False], ["norm", "component", "integral"])),
    )
    def test_e3nn_parameters(xyz, normalize, normalization):
        """Checks that the different normalization options match."""

        l_list = list(range(10))
        e3nn_reference = e3nn.o3.spherical_harmonics(
            l_list, xyz, normalize, normalization
        )
        sh = sphericart.torch.e3nn_spherical_harmonics(
            l_list, xyz, normalize, normalization
        )

        assert relative_mse(e3nn_reference.detach(), sh.detach()) < TOLERANCE

    def test_e3nn_patch(xyz):
        """Tests the patch function."""
        e3nn_reference = e3nn.o3.spherical_harmonics([1, 3, 5], xyz, True)
        e3nn_builtin = e3nn.o3.spherical_harmonics

        sphericart.torch.patch_e3nn(e3nn)

        assert e3nn.o3.spherical_harmonics is sphericart.torch.e3nn_spherical_harmonics
        sh = e3nn.o3.spherical_harmonics([1, 3, 5], xyz, True)

        # restore spherical_harmonics
        sphericart.torch.unpatch_e3nn(e3nn)
        assert e3nn.o3.spherical_harmonics is e3nn_builtin

        assert relative_mse(e3nn_reference.detach(), sh.detach()) < TOLERANCE
