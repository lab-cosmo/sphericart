import pytest
import torch

import sphericart.torch


torch.manual_seed(0)


class SHModule(torch.nn.Module):
    """Example of how to use SphericalHarmonics from within a
    `torch.nn.Module`"""

    def __init__(self, l_max, normalized=False):
        super().__init__()
        if normalized:
            self._sph = sphericart.torch.SphericalHarmonics(l_max)
        else:
            self._sph = sphericart.torch.SolidHarmonics(l_max)

    def forward(self, xyz):
        sph = self._sph.compute(xyz)
        return sph


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("normalized", [True, False])
def test_script(xyz, normalized):
    xyz_jit = xyz.detach().clone().requires_grad_()
    module = SHModule(l_max=10, normalized=normalized)
    sh_module = module.forward(xyz)
    sh_module.sum().backward()

    # JIT compilation of the module
    script = torch.jit.script(module)
    sh_script = script.forward(xyz_jit)
    sh_script.sum().backward()
