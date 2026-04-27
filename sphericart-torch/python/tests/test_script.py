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


def _loss(module, xyz):
    sph = module(xyz)[0]
    return (
        sph[1] ** 2
        + sph[2]
        + 0.20 * sph[1] * sph[3]
        + sph[2] ** 2
        + 0.42 * sph[1] * sph[2]
        + sph[0] ** 2
    )


@pytest.mark.parametrize("normalized", [True, False])
def test_script(xyz, normalized):
    xyz_eager = xyz.detach().clone().requires_grad_()
    xyz_jit = xyz.detach().clone().requires_grad_()
    module = SHModule(l_max=10, normalized=normalized)
    sh_module = module.forward(xyz_eager)
    sh_module.sum().backward()

    # JIT compilation of the module
    script = torch.jit.script(module)
    sh_script = script.forward(xyz_jit)
    sh_script.sum().backward()

    assert torch.allclose(sh_script, sh_module)
    assert torch.allclose(xyz_jit.grad, xyz_eager.grad)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        xyz_jit_cuda = xyz.detach().clone().to("cuda").requires_grad_()

        sh_cuda = module.forward(xyz_cuda)
        sh_cuda.sum().backward()
        sh_jit_cuda = script.forward(xyz_jit_cuda)
        sh_jit_cuda.sum().backward()

        assert torch.allclose(sh_jit_cuda, sh_cuda)
        assert torch.allclose(xyz_jit_cuda.grad, xyz_cuda.grad)


@pytest.mark.parametrize("normalized", [True, False])
def test_script_save_load(xyz, normalized, tmp_path):
    module = SHModule(l_max=10, normalized=normalized)
    scripted = torch.jit.script(module)
    path = tmp_path / "module.pt"
    scripted.save(path)

    loaded = torch.jit.load(path)
    expected = scripted(xyz.detach())
    actual = loaded(xyz.detach())

    assert torch.allclose(actual, expected)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().to("cuda")
        assert torch.allclose(loaded(xyz_cuda), scripted(xyz_cuda))


@pytest.mark.parametrize("normalized", [True, False])
def test_script_hessians(normalized):
    module = SHModule(l_max=1, normalized=normalized)
    script = torch.jit.script(module)
    xyz = torch.tensor([[0.67, 0.53, -0.22]], dtype=torch.float64, requires_grad=True)
    xyz_jit = xyz.detach().clone().requires_grad_()

    grad = torch.autograd.grad(_loss(module, xyz), xyz, create_graph=True)[0]
    grad_jit = torch.autograd.grad(_loss(script, xyz_jit), xyz_jit, create_graph=True)[
        0
    ]
    assert torch.allclose(grad_jit, grad)

    hessian = torch.autograd.functional.hessian(lambda x: _loss(module, x), xyz)
    hessian_jit = torch.autograd.functional.hessian(lambda x: _loss(script, x), xyz_jit)
    assert torch.allclose(hessian_jit, hessian)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        xyz_jit_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        grad_cuda = torch.autograd.grad(
            _loss(module, xyz_cuda), xyz_cuda, create_graph=True
        )[0]
        grad_jit_cuda = torch.autograd.grad(
            _loss(script, xyz_jit_cuda), xyz_jit_cuda, create_graph=True
        )[0]
        assert torch.allclose(grad_jit_cuda, grad_cuda)
        hessian_cuda = torch.autograd.functional.hessian(
            lambda x: _loss(module, x), xyz_cuda
        )
        hessian_jit_cuda = torch.autograd.functional.hessian(
            lambda x: _loss(script, x), xyz_jit_cuda
        )
        assert torch.allclose(hessian_jit_cuda, hessian_cuda)
