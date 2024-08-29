import torch
import sphericart.torch

xyz = torch.randn(10, 3, device='cuda', requires_grad=True)
xyz_cpu = xyz.clone().detach().cpu().requires_grad_(True)

sph = sphericart.torch.SphericalHarmonics(3)

print("-- result (CPU) --")
ylm_cpu = sph.compute(xyz_cpu)
print(ylm_cpu)

ylm_cpu.sum().backward()

print(xyz_cpu.grad)
print(xyz_cpu.dtype)

print("-- result (CUDA) --")
ylm = sph.compute(xyz)
print(ylm)

ylm_2 = sph.compute(xyz)
print(ylm_2)

ylm.sum().backward()

#torch.cuda.synchronize()

print(xyz.grad.cpu() - xyz_cpu.grad)
