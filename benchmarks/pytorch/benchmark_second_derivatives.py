import argparse
import time

import numpy as np
import torch

import sphericart.torch


docstring = """
Benchmarks for the torch second derivatives in `sphericart`.

Compares with e3nn if present and if the comparison is
requested.
"""

try:
    import e3nn
    from e3nn import o3

    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False

try:
    import jax

    jax.config.update("jax_enable_x64", True)  # enable float64 for jax
    import e3nn_jax
    import jax.numpy as jnp

    _HAS_E3NN_JAX = True
except ImportError:
    _HAS_E3NN_JAX = False


def sphericart_benchmark(
    l_max=10,
    n_samples=100,
    n_tries=100,
    normalized=False,
    device="cpu",
    dtype=torch.float64,
    compare=False,
    verbose=False,
    warmup=16,
):
    xyz = torch.randn((n_samples, 3), dtype=dtype, device=device, requires_grad=True)
    sh_calculator = sphericart.torch.SphericalHarmonics(l_max, normalized=normalized, backward_second_derivatives=True)
    omp_threads = sh_calculator.omp_num_threads()
    print(
        f"**** Timings for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, "
        + f"dtype={dtype}, device={device}, omp_num_threads={omp_threads} ****"
    )

    time_grad_grad = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = sh_calculator.compute(xyz)
        sph_sum = torch.sum(sh_sphericart)
        grad = torch.autograd.grad(
            sph_sum,
            xyz,
            retain_graph=True,
            create_graph=True
        )[0]
        grad_grad = torch.autograd.grad(
            torch.sum(grad),
            xyz
        )[0]
        elapsed += time.time()
        time_grad_grad[i] = elapsed

    mean_time = time_grad_grad[warmup:].mean() / n_samples
    std_time = time_grad_grad[warmup:].std() / n_samples
    print(
        f" sphericart gradgrad:  {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_hessian[:warmup])

    def function_sphericart(xyz):
        sh_sphericart = sh_calculator.compute(xyz)
        return torch.sum(sh_sphericart)

    time_hessian = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        hessian = torch.autograd.functional.hessian(function_sphericart, xyz, vectorize=True)
        elapsed += time.time()
        time_hessian[i] = elapsed

    mean_time = time_hessian[warmup:].mean() / n_samples
    std_time = time_hessian[warmup:].std() / n_samples
    print(
        f" sphericart Hessian:   {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_hessian[:warmup])

    if compare and _HAS_E3NN:
        xyz = torch.randn((n_samples, 3), dtype=dtype, device=device, requires_grad=True)

        sh_e3nn = e3nn.o3.spherical_harmonics(
            list(range(l_max + 1)),
            xyz,
            normalize=normalized,
            normalization="integral",
        )
        sph_sum = torch.sum(sh_e3nn)

        for i in range(n_tries + warmup):
            elapsed = -time.time()
            sh_e3nn = e3nn.o3.spherical_harmonics(
                list(range(l_max + 1)),
                xyz,
                normalize=normalized,
                normalization="integral",
            )
            sph_sum = torch.sum(sh_e3nn)
            grad = torch.autograd.grad(
                sph_sum,
                xyz,
                retain_graph=True,
                create_graph=True
            )[0]
            grad_grad = torch.autograd.grad(
                torch.sum(grad),
                xyz
            )[0]
            elapsed += time.time()
            time_grad_grad[i] = elapsed

        mean_time = time_grad_grad[warmup:].mean() / n_samples
        std_time = time_grad_grad[warmup:].std() / n_samples
        print(
            f" E3NN gradgrad:        {mean_time * 1e9: 10.1f} ns/sample ± "
            + f"{std_time * 1e9: 10.1f} (std)"
        )
        if verbose:
            print("Warm-up timings / sec.: \n", time_hessian[:warmup])

        def function_e3nn(xyz):
            sh_e3nn = o3.spherical_harmonics(
                list(range(l_max + 1)),
                xyz,
                normalize=normalized,
                normalization="integral",
            )
            return torch.sum(sh_e3nn)

        for i in range(n_tries + warmup):
            elapsed = -time.time()
            hessian = torch.autograd.functional.hessian(function_e3nn, xyz)
            elapsed += time.time()
            time_hessian[i] = elapsed

        mean_time = time_hessian[warmup:].mean() / n_samples
        std_time = time_hessian[warmup:].std() / n_samples
        print(
            f" E3NN Hessian:         {mean_time * 1e9: 10.1f} ns/sample ± "
            + f"{std_time * 1e9: 10.1f} (std)"
        )
        if verbose:
            print("Warm-up timings / sec.: \n", time_hessian[:warmup])

    print(
        "******************************************************************************"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=3, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=100, help="number of samples")
    parser.add_argument("-t", type=int, default=100, help="number of runs/sample")
    parser.add_argument(
        "-cpu", type=int, default=1, help="print CPU results (0=False, 1=True)"
    )
    parser.add_argument(
        "-gpu", type=int, default=1, help="print GPU results (0=False, 1=True)"
    )
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose timing output",
    )
    parser.add_argument(
        "--warmup", type=int, default=16, help="number of warm-up evaluations"
    )

    args = parser.parse_args()

    # Run benchmarks
    if args.cpu:
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cpu",
            dtype=torch.float64,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cpu",
            dtype=torch.float32,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )

    if torch.cuda.is_available() and args.gpu:
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float64,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float32,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )
