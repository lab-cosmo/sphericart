import pytest
import torch
import torch._dynamo as dynamo

import sphericart.torch


torch.manual_seed(0)


class SHModule(torch.nn.Module):
    def __init__(self, l_max, normalized=False):
        super().__init__()
        if normalized:
            self._sph = sphericart.torch.SphericalHarmonics(l_max)
        else:
            self._sph = sphericart.torch.SolidHarmonics(l_max)

    def forward(self, xyz):
        return self._sph.compute(xyz)


@pytest.fixture
def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)


def _single(module, xyz):
    return module(xyz.unsqueeze(0)).squeeze(0)


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

    if torch.cuda.is_available():
        xyz_eager_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        xyz_compiled_cuda = xyz.detach().clone().to("cuda").requires_grad_()

        sh_eager_cuda = module(xyz_eager_cuda)
        sh_eager_cuda.sum().backward()
        eager_grad_cuda = xyz_eager_cuda.grad.detach().clone()

        sh_compiled_cuda = compiled(xyz_compiled_cuda)
        sh_compiled_cuda.sum().backward()

        assert torch.allclose(sh_compiled_cuda, sh_eager_cuda)
        assert torch.allclose(xyz_compiled_cuda.grad, eager_grad_cuda)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_grad(xyz, normalized):
    module = SHModule(l_max=10, normalized=normalized)
    xyz_eager = xyz.detach().clone().requires_grad_()
    xyz_compiled = xyz.detach().clone().requires_grad_()

    eager_grad = torch.autograd.grad(module(xyz_eager).sum(), xyz_eager)[0]
    compiled_grad = torch.autograd.grad(
        torch.compile(module, fullgraph=True)(xyz_compiled).sum(), xyz_compiled
    )[0]
    assert torch.allclose(compiled_grad, eager_grad)

    if torch.cuda.is_available():
        xyz_eager_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        xyz_compiled_cuda = xyz.detach().clone().to("cuda").requires_grad_()
        eager_grad_cuda = torch.autograd.grad(
            module(xyz_eager_cuda).sum(), xyz_eager_cuda
        )[0]
        compiled_grad_cuda = torch.autograd.grad(
            torch.compile(module, fullgraph=True)(xyz_compiled_cuda).sum(),
            xyz_compiled_cuda,
        )[0]
        assert torch.allclose(compiled_grad_cuda, eager_grad_cuda)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_vmap(xyz, normalized):
    module = SHModule(l_max=4, normalized=normalized)
    eager = torch.func.vmap(lambda x: _single(module, x))(xyz.detach())
    compiled = torch.compile(
        torch.func.vmap(lambda x: _single(module, x)), fullgraph=True
    )(xyz.detach())
    assert torch.allclose(compiled, eager)

    if torch.cuda.is_available():
        xyz_cuda = xyz.detach().to("cuda")
        eager_cuda = torch.func.vmap(lambda x: _single(module, x))(xyz_cuda)
        compiled_cuda = torch.compile(
            torch.func.vmap(lambda x: _single(module, x)), fullgraph=True
        )(xyz_cuda)
        assert torch.allclose(compiled_cuda, eager_cuda)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_vmap_noncontiguous(normalized):
    module = SHModule(l_max=4, normalized=normalized)
    xyz = torch.randn(3, dtype=torch.float64)
    batch = xyz.expand(5, -1)

    eager = torch.func.vmap(lambda x: _single(module, x))(batch)
    compiled = torch.compile(
        torch.func.vmap(lambda x: _single(module, x)), fullgraph=True
    )(batch)
    assert torch.allclose(compiled, eager)

    if torch.cuda.is_available():
        batch_cuda = batch.to("cuda")
        eager_cuda = torch.func.vmap(lambda x: _single(module, x))(batch_cuda)
        compiled_cuda = torch.compile(
            torch.func.vmap(lambda x: _single(module, x)), fullgraph=True
        )(batch_cuda)
        assert torch.allclose(compiled_cuda, eager_cuda)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_jacfwd(normalized):
    module = SHModule(l_max=4, normalized=normalized)
    xyz = torch.tensor([[0.67, 0.53, -0.22]], dtype=torch.float64)
    eager = torch.func.jacfwd(lambda x: _single(module, x))(xyz[0])
    compiled = torch.compile(
        torch.func.jacfwd(lambda x: _single(module, x)), fullgraph=True
    )(xyz[0])
    assert torch.allclose(compiled, eager)

    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")
        eager_cuda = torch.func.jacfwd(lambda x: _single(module, x))(xyz_cuda[0])
        compiled_cuda = torch.compile(
            torch.func.jacfwd(lambda x: _single(module, x)), fullgraph=True
        )(xyz_cuda[0])
        assert torch.allclose(compiled_cuda, eager_cuda)


@pytest.mark.parametrize("normalized", [True, False])
def test_compile_has_no_graph_breaks(xyz, normalized):
    module = SHModule(l_max=10, normalized=normalized)
    explanation = dynamo.explain(module)(xyz.detach())

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []
