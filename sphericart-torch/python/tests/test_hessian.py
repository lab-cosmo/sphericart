import pytest
import torch

import sphericart.torch


xyz = 6 * torch.randn(20, 3, dtype=torch.float64, requires_grad=True)


calculator = sphericart.torch.SphericalHarmonics(
    l_max=5, normalized=True, backward_second_derivatives=True
)

_, grads_cpu, hess_cpu = calculator.compute_with_hessians(xyz)

xyz_cuda = xyz.to(device="cuda")
_, grads_cuda, hess_cuda = calculator.compute_with_hessians(xyz_cuda)


print (torch.allclose(grads_cpu.cuda(), grads_cuda, atol=1e-5))
print (torch.allclose(hess_cpu.cuda(), hess_cuda, atol=1e-5))