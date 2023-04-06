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


def sphericart_benchmark(
    l_max=10,
    n_samples=10000,
    n_tries=100,
    normalized=False,
    device="cpu",
    dtype=torch.float64,
    compare=False,
):
    print(
        f" ** Timing for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, dtype={dtype}, device={device}"
    )
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

    print(
        f" No derivatives: {time_noderi.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_noderi.std()/n_samples*1e9: 10.1f} (std)"
    )

    sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)

    time_deri = np.zeros(n_tries)
    for i in range(n_tries):
        elapsed = -time.time()
        sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)
        elapsed += time.time()
        time_deri[i] = elapsed

    print(
        f" Derivatives:    {time_deri.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_deri.std()/n_samples*1e9: 10.1f} (std)"
    )

    # autograd
    xyz = xyz.clone().detach().type(dtype).to(device).requires_grad_()
    sh_sphericart, _ = sh_calculator.compute(xyz)

    sph_sum = torch.sum(sh_sphericart)
    sph_sum.backward()

    time_fw = np.zeros(n_tries)
    time_bw = np.zeros(n_tries)

    for i in range(n_tries):
        elapsed = -time.time()
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
        elapsed += time.time()
        time_fw[i] = elapsed

        sph_sum = torch.sum(sh_sphericart)
        elapsed = -time.time()
        sph_sum.backward()
        elapsed += time.time()
        time_bw[i] = elapsed

    print(
        f" Autograd:       {time_fw.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw.std()/n_samples*1e9: 10.1f} (std)"
    )
    print(
        f" Backprop:       {time_bw.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw.std()/n_samples*1e9: 10.1f} (std)"
    )

    if compare and _HAS_E3NN:
        xyz_tensor = (
            xyz[:, [1, 2, 0]].clone().detach().type(dtype).to(device).requires_grad_()
        )
        sh = e3nn.o3.spherical_harmonics(
            list(range(l_max + 1)), xyz_tensor, normalize=normalized
        )
        for i in range(n_tries):
            elapsed = -time.time()
            sh_e3nn = e3nn.o3.spherical_harmonics(
                list(range(l_max + 1)), xyz_tensor, normalize=normalized
            )
            elapsed += time.time()
            time_fw[i] = elapsed
            sph_sum = torch.sum(sh_e3nn)
            elapsed = -time.time()
            sph_sum.backward()
            elapsed += time.time()
            time_bw[i] = elapsed

        print(
            f" E3NN-FW:        {time_fw.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw.std()/n_samples*1e9: 10.1f} (std)"
        )
        print("First-calls timings / sec.: \n", time_fw[:4])
        print(
            f" E3NN-BW:        {time_bw.mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw.std()/n_samples*1e9: 10.1f} (std)"
        )
        print("First-calls timings / sec.: \n", time_bw[:4])
    print("****************************************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=1000, help="number of samples")
    parser.add_argument("-t", type=int, default=100, help="number of runs/sample")
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="compute normalized spherical harmonics",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="compare timings with other codes, if installed",
    )

    args = parser.parse_args()

    # Run benchmarks
    sphericart_benchmark(
        args.l,
        args.s,
        args.t,
        args.normalized,
        device="cpu",
        dtype=torch.float64,
        compare=args.compare,
    )
    sphericart_benchmark(
        args.l,
        args.s,
        args.t,
        args.normalized,
        device="cpu",
        dtype=torch.float32,
        compare=args.compare,
    )

    if torch.cuda.is_available():
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float64,
            compare=args.compare,
        )
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float32,
            compare=args.compare,
        )
