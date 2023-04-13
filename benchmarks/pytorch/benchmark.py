import argparse
import time

import numpy as np

import sphericart_torch
import torch

docstring = """
Benchmarks for the torch implementation of ``sphericart``.
Compares with e3nn and e3nn_jax if those are present 
and if the comparison is requested.
"""

try:
    import e3nn
    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False

try:
    import jax
    jax.config.update("jax_enable_x64", True)  # enable float64 for jax
    import jax.numpy as jnp
    import e3nn_jax
    _HAS_E3NN_JAX = True
except ImportError:
    _HAS_E3NN_JAX = False


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
        f"**** Timings for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, dtype={dtype}, device={device} ****"
    )
    xyz = torch.randn((n_samples, 3), dtype=dtype, device=device)
    sh_calculator = sphericart_torch.SphericalHarmonics(l_max, normalized=normalized)

    time_noderi = np.zeros(n_tries+10)
    for i in range(n_tries+10):
        elapsed = -time.time()
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
        elapsed += time.time()
        time_noderi[i] = elapsed

    print(
        f" No derivatives: {time_noderi[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_noderi[10:].std()/n_samples*1e9: 10.1f} (std)"
    )

    time_deri = np.zeros(n_tries+10)
    for i in range(n_tries+10):
        elapsed = -time.time()
        sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)
        elapsed += time.time()
        time_deri[i] = elapsed

    print(
        f" Derivatives:    {time_deri[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_deri[10:].std()/n_samples*1e9: 10.1f} (std)"
    )

    # autograd
    xyz = xyz.clone().detach().type(dtype).to(device).requires_grad_()

    time_fw = np.zeros(n_tries+10)
    time_bw = np.zeros(n_tries+10)

    for i in range(n_tries+10):
        elapsed = -time.time()
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
        elapsed += time.time()
        time_fw[i] = elapsed

        sph_sum = torch.sum(sh_sphericart)
        elapsed = -time.time()
        sph_sum.backward()
        elapsed += time.time()
        time_bw[i] = elapsed

    # print(xyz.grad)

    print(
        f" Forward:        {time_fw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw[10:].std()/n_samples*1e9: 10.1f} (std)"
    )
    print(
        f" Backward:       {time_bw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw[10:].std()/n_samples*1e9: 10.1f} (std)"
    )

    if compare and _HAS_E3NN:
        xyz_tensor = (
            xyz[:, [1, 2, 0]].clone().detach().type(dtype).to(device).requires_grad_()
        )
        for i in range(n_tries+10):
            elapsed = -time.time()
            sh_e3nn = e3nn.o3.spherical_harmonics(
                list(range(l_max + 1)), xyz_tensor, normalize=normalized, normalization="integral"
            )
            elapsed += time.time()
            time_fw[i] = elapsed
            sph_sum = torch.sum(sh_e3nn)
            elapsed = -time.time()
            sph_sum.backward()
            elapsed += time.time()
            time_bw[i] = elapsed

        # print(xyz_tensor.grad)

        print(
            f" E3NN-FW:        {time_fw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw[10:].std()/n_samples*1e9: 10.1f} (std)"
        )
        # print("First-calls timings / sec.: \n", time_fw[:10])
        print(
            f" E3NN-BW:        {time_bw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw[10:].std()/n_samples*1e9: 10.1f} (std)"
        )
        # print("First-calls timings / sec.: \n", time_bw[:10])

    if compare and _HAS_E3NN_JAX:
        dtype = (np.float64 if dtype == torch.float64 else np.float32)
        xyz_tensor = jnp.asarray(
            xyz[:, [1, 2, 0]].clone().detach().cpu().numpy(), dtype=dtype
        )  # Automatically goes to gpu if present
        if device == "cpu": xyz_tensor = jax.device_put(xyz_tensor, jax.devices("cpu")[0])  # Force back to cpu

        irreps = e3nn_jax.Irreps([e3nn_jax.Irrep(l, 1) for l in range(l_max+1)])

        def loss_fn(xyz_tensor):
            sh_e3nn = e3nn_jax.spherical_harmonics(
                irreps, xyz_tensor, normalize=normalized, normalization="integral"
            )
            loss = jnp.sum(sh_e3nn.array)
            return loss
        
        loss_grad_fn = jax.grad(loss_fn)

        for i in range(n_tries+10):
            elapsed = -time.time()
            loss = loss_fn(xyz_tensor)
            elapsed += time.time()
            time_fw[i] = elapsed

            elapsed = -time.time()
            loss_grad = loss_grad_fn(xyz_tensor)
            elapsed += time.time()
            time_bw[i] = elapsed

        # print(loss_grad)

        print(
            f" E3NN-JAX-FW:    {time_fw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw[10:].std()/n_samples*1e9: 10.1f} (std)"
        )
        # print("First-calls timings / sec.: \n", time_fw[:10])
        print(
            f" E3NN-JAX-BW:    {time_bw[10:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw[10:].std()/n_samples*1e9: 10.1f} (std)"
        )
        # print("First-calls timings / sec.: \n", time_bw[:10])
    print("*********************************************************************************************")


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
