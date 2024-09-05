import torch
import sphericart.torch
from time import time
xyz = torch.rand(10, 3, device='cuda', requires_grad=True, dtype=torch.float32)
xyz_cpu = xyz.clone().detach().cpu().requires_grad_(True)

sph = sphericart.torch.SphericalHarmonics(8)

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

print(ylm_2 - ylm)

# ylm.sum().backward()

sph = sphericart.torch.SphericalHarmonics(8)
xyz = torch.rand(100000, 3, device='cuda',
                 requires_grad=False, dtype=torch.float32)

# warmup
torch.cuda.synchronize()
for i in range(100):
    ylm = sph.compute(xyz)

torch.cuda.synchronize()
start = time()

for i in range(1000):
    ylm = sph.compute(xyz)
torch.cuda.synchronize()
print(time() - start, "(s)")

# print(xyz.grad.cpu() - xyz_cpu.grad)
