import argparse
import time

import numpy as np 

import jax
import sphericart.jax

jax.config.update("jax_enable_x64", True)  # enable float64 for jax

docstring = """
Benchmarks for the jax implementation of `sphericart`.

Compares with e3nn_jax if present and if the comparison is requested.
"""

try:
    import e3nn_jax
    import jax.numpy as jnp

    _HAS_E3NN_JAX = True
except ImportError:
    _HAS_E3NN_JAX = False


def sphericart_benchmark(
    l_max=10,
    n_samples=10000,
    n_tries=100,
    normalized=False,
    device="cpu",
    dtype=np.float64,
    compare=False,
    verbose=False,
    warmup=16,
):
    key = jax.random.PRNGKey(0)
    xyz = jax.random.normal(key, (n_samples, 3), dtype=dtype)

    sh_calculator = sphericart.jax.spherical_harmonics

    sh_calculator_jit = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1)

    print(
        f"**** Timings for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, "
        + f"dtype={dtype} ****"
    )
    
    time_noderi = np.zeros(n_tries + warmup)
    time_fw = np.zeros(n_tries + warmup)
    time_bw = np.zeros(n_tries + warmup)
    
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = sh_calculator(xyz, l_max, normalized)
        elapsed += time.time()
        time_noderi[i] = elapsed

    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" No derivatives: {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_noderi[:warmup])

    time_noderi[:] = 0.0

    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart_jit = sh_calculator_jit(xyz, l_max, normalized)
        elapsed += time.time()
        time_noderi[i] = elapsed

    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" No derivatives, jit: {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_noderi[:warmup])


    if compare and _HAS_E3NN_JAX:
        xyz_tensor = xyz.copy()

        irreps = e3nn_jax.Irreps([e3nn_jax.Irrep(l, 1) for l in range(l_max + 1)])

        def loss_fn(xyz_tensor):
            sh_e3nn = e3nn_jax.spherical_harmonics(
                irreps, xyz_tensor, normalize=normalized, normalization="integral"
            )
            loss = jnp.sum(sh_e3nn.array)
            return loss

        loss_grad_fn = jax.grad(loss_fn)

        for i in range(n_tries + warmup):
            elapsed = -time.time()
            _ = loss_fn(xyz_tensor)
            elapsed += time.time()
            time_fw[i] = elapsed

            elapsed = -time.time()
            _ = loss_grad_fn(xyz_tensor)
            elapsed += time.time()
            time_bw[i] = elapsed

        mean_time = time_fw[warmup:].mean() / n_samples
        std_time = time_fw[warmup:].std() / n_samples
        print(
            f" E3NN-JAX-FW:    {mean_time*1e9: 10.1f} ns/sample ± "
            + f"{std_time*1e9: 10.1f} (std)"
        )
        if verbose:
            print("Warm-up timings / sec.: \n", time_fw[:warmup])
        mean_time = time_bw[warmup:].mean() / n_samples
        std_time = time_bw[warmup:].std() / n_samples
        print(
            f" E3NN-JAX-BW:    {mean_time*1e9: 10.1f} ns/sample ± "
            + f"{std_time*1e9: 10.1f} (std)"
        )
        if verbose:
            print("Warm-up timings / sec.: \n", time_bw[:warmup])
    print(
        "******************************************************************************"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=10000, help="number of samples")
    parser.add_argument("-t", type=int, default=100, help="number of runs/sample")
    parser.add_argument(
        "-cpu", type=int, default=1, help="print CPU results (0=False, 1=True)"
    )
    parser.add_argument(
        "-gpu", type=int, default=0, help="print GPU results (0=False, 1=True)"
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
            dtype=np.float64,
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
            dtype=np.float32,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )

    if args.gpu:
        raise ValueError("GPU implementation of sphericart-jax is not yet available")
