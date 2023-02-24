import numpy as np
import sphericart
import time

docstring = """
An example of the use of the Python interface of the `sphericart` library.

Simply computes Cartesian spherical harmonics for the given parameters, for an 
array of random 3D points. Also gets some timing information. 
If the `e3nn` package is present, also compares results.
"""

try:
    import torch
    import e3nn
    _HAS_E3NN = True
except:
    _HAS_E3NN = False

def sphericart_example(l_max=10, n_samples=10000, n_tries=100, normalized=False):
    
    xyz = np.random.rand(n_samples, 3)
    
    print("Timings")

    sh_calculator = sphericart.SphericalHarmonics(l_max, normalized=normalized)
    start = time.time()
    for _ in range(n_tries):
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
    finish = time.time()
    print(f"sphericart took {1e9*(finish-start)/n_tries/n_samples} ns/sample")

    if _HAS_E3NN:
        xyz_tensor = torch.tensor(xyz)
        sh = e3nn.o3.spherical_harmonics(l_max, xyz_tensor, normalize=normalized)  # allow compilation (??)
        start = time.time()
        for _ in range(n_tries):
            sh = e3nn.o3.spherical_harmonics(l_max, xyz_tensor, normalize=normalized)
        finish = time.time()
        print(f"e3nn took {1e9*(finish-start)/n_tries/n_samples} ns/sample")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="Maximum angular momentum.")
    parser.add_argument("-t", type=int, default=100, help=".")
    parser.add_argument("-s", type=int, default=1000, help="Step to end.")
    parser.add_argument(
        "--normalized", action="store_true", default=False, help="Wrap atomic positions."
    )

    args = parser.parse_args()

    # Process everything.
    sphericart_example(
        args.l,
        args.s,
        args.t,
        args.normalized
    )

"""
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
"""