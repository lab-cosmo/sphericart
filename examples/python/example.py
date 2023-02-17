import numpy as np
import sphericart
import torch
import e3nn
import time


l_max = 10
n_samples = 10000
n_tries = 100
xyz = np.random.rand(n_samples, 3)


print("Timings")

sh_calculator = sphericart.SphericalHarmonics(l_max)
start = time.time()
for _ in range(n_tries):
    sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
finish = time.time()
print(f"sphericart took {1000*(finish-start)/n_tries} ms")

xyz_tensor = torch.tensor(xyz)
sh = e3nn.o3.spherical_harmonics(l_max, xyz_tensor, normalize=False)  # allow compilation (??)
start = time.time()
for _ in range(100):
    sh = e3nn.o3.spherical_harmonics(l_max, xyz_tensor, normalize=False)
finish = time.time()
print(f"e3nn took {1000*(finish-start)/n_tries} ms")


print("Derivative timings")

start = time.time()
for _ in range(100):
    sh_sphericart, sh_derivatives = sh_calculator.compute(xyz, gradients=True)
    dummy_loss = sh_sphericart.sum()
    loss_derivatives  = sh_derivatives.sum(axis=2)
finish = time.time()
print(f"sphericart took {1000*(finish-start)/n_tries} ms")

xyz_tensor.requires_grad = True
start = time.time()
for _ in range(100):
    sh = e3nn.o3.spherical_harmonics(l_max, xyz_tensor, normalize=False)
    dummy_loss = torch.sum(sh)
    dummy_loss.backward()
finish = time.time()
print(f"e3nn took {1000*(finish-start)/n_tries} ms")
