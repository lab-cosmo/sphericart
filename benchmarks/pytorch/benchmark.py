import argparse
import time

import numpy as np

import sphericart_torch
import torch

docstring = """
Benchmarks for the torch implementation of ``sphericart``.
Compares with E3NN if present.
"""

try:
    import e3nn

    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False

def sphericart_benchmark(l_max=10, n_samples=10000, n_tries=100, normalized=False, device="cpu", dtype=torch.float64):
    
    xyz = torch.randn((n_samples, 3), dtype=dtype, device=device)
    sh_calculator = sphericart_torch.SphericalHarmonics(l_max, normalized=normalized)
    
    # one call to compile the kernels
    sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)

    time_noderi = np.zeros(n_tries)
    for i in range(n_tries):
        elapsed = -time.time()
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
        elapsed += time.time()
        time_noderi[i] = elapsed

    sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)

    time_deri = np.zeros(n_tries)
    for i in range(n_tries):
        elapsed = -time.time()
        sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)
        elapsed += time.time()
        time_deri[i] = elapsed

    # autograd
    xyz = xyz.clone().detach().type(dtype).to(device).requires_grad_()
    sh_sphericart, _ = sh_calculator.compute(xyz)

    sph_norm = torch.sum(sh_sphericart**2)
    sph_norm.backward()

    time_fw = np.zeros(n_tries)
    time_bw = np.zeros(n_tries)

    for i in range(n_tries):
        elapsed = -time.time()
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
        elapsed += time.time()
        time_fw[i] = elapsed

        sph_norm = torch.sum(sh_sphericart**2)
        elapsed = -time.time()
        sph_norm.backward()
        elapsed += time.time()
        time_bw[i] = elapsed

    print(f" ** Timing for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, dtype={dtype}, device={device}")
    print(f" No derivatives: {time_noderi.mean()/n_samples*1e9} ns")
    print(f" Derivatives:    {time_deri.mean()/n_samples*1e9} ns")
    print(f" Autograd:       {time_fw.mean()/n_samples*1e9} ns")
    print(f" Backprop:       {time_bw.mean()/n_samples*1e9} ns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=1000, help="number of samples")
    parser.add_argument("-t", type=int, default=100, help="number of runs")
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="compute normalized spherical harmonics",
    )

    args = parser.parse_args()

    # Process everything.
    sphericart_benchmark(args.l, args.s, args.t, args.normalized, device="cpu", dtype=torch.float64)