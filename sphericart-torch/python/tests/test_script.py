import pytest
import torch
import torch._dynamo as dynamo

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


@pytest.mark.parametrize("normalized", [True, False])
def test_compile(xyz, normalized):
    xyz_eager = xyz.detach().clone().requires_grad_()
    xyz_compiled = xyz.detach().clone().requires_grad_()
    module = SHModule(l_max=10, normalized=normalized)

    sh_eager = module(xyz_eager)
    sh_eager.sum().backward()
    eager_grad = xyz_eager.grad.detach().clone()

    compiled = torch.compile(module, fullgraph=True)
    sh_compiled = compiled(xyz_compiled)
    sh_compiled.sum().backward()

    assert torch.allclose(sh_compiled, sh_eager)
    assert torch.allclose(xyz_compiled.grad, eager_grad)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_has_no_graph_breaks(xyz, normalized):
    module = SHModule(l_max=10, normalized=normalized)
    explanation = dynamo.explain(module)(xyz.detach())

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []
