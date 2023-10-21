import argparse
import time

import numpy as np

import jax
import sphericart.jax

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
    n_samples=200,
    n_tries=100,
    normalized=False,
    device="cpu",
    dtype=jnp.float64,
    compare=False,
    verbose=False,
    warmup=16,
):

    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)  # enable float64 for jax
    else:
        jax.config.update("jax_enable_x64", False)  # disenable float64 for jax

    key = jax.random.PRNGKey(0)
    xyz = jax.random.normal(key, (n_samples, 3), dtype=dtype)

    sh_calculator = sphericart.jax.spherical_harmonics

    sh_calculator_jit = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1)

    print(
        f"**** Timings for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, "
        + f"dtype={dtype} ****"
    )

    time_noderi = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = sh_calculator(xyz, l_max, normalized)
        elapsed += time.time()
        time_noderi[i] = elapsed
    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" No derivatives:      {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_noderi[:warmup])

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
    if verbose: print("Warm-up timings / sec.:\n", time_noderi[:warmup])

    def scalar_output(xyz, l_max, normalized):
        return jnp.sum(sphericart.jax.spherical_harmonics(xyz, l_max, normalized))

    sh_grad = jax.jit(jax.grad(scalar_output), static_argnums=1)

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart_grad_jit = sh_grad(xyz, l_max, normalized)
        elapsed += time.time()
        time_deri[i] = elapsed

    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" grad (jit):          {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    def single_scalar_output(x, l_max, normalized):
        return jnp.sum(sphericart.jax.spherical_harmonics(x, l_max, normalized))

    # Compute the Hessian for a single (3,) input
    single_hessian = jax.hessian(single_scalar_output)

    # Use vmap to vectorize the Hessian computation over the first axis
    sh_hess = jax.jit(
        jax.vmap(single_hessian, in_axes=(0, None, None)), static_argnums=1
    )

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart_hess_jit = sh_hess(xyz, l_max, normalized)
        elapsed += time.time()
        time_deri[i] = elapsed

    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" hessian (jit):       {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    # calculate a function of the spherical harmonics that returns an array
    # and take its jacobian with respect to the input Cartesian coordinates,
    # both in forward mode and in reverse mode
    def array_output(xyz, l_max, normalized):
        return jnp.sum(
            sphericart.jax.spherical_harmonics(xyz, l_max, normalized), axis=0
        )

    jacfwd = jax.jit(jax.jacfwd(array_output), static_argnums=1)

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_jacfwd_jit = jacfwd(xyz, l_max, normalized)
        elapsed += time.time()
        time_deri[i] = elapsed

    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" jacfwd (jit):        {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    jacrev = jax.jit(jax.jacrev(array_output), static_argnums=1)

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_jacrev_jit = jacrev(xyz, l_max, normalized)
        elapsed += time.time()
        time_deri[i] = elapsed

    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" jacrev (jit):        {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    if compare and _HAS_E3NN_JAX:
        # compare to e3nn-jax
        irreps = e3nn_jax.Irreps([e3nn_jax.Irrep(l, 1) for l in range(l_max + 1)])

        def e3nn_sph(xyz):
            return e3nn_jax.spherical_harmonics(
                irreps, xyz, normalize=normalized, normalization="integral"
            ).array

        def single_scalar_output(xyz):
            sh_e3nn = e3nn_sph(xyz)
            loss = jnp.sum(sh_e3nn)
            return loss

        def array_output(xyz):
            return jnp.sum(
                e3nn_sph(xyz), axis=0
            )

        jit_e3nn_sph = jax.jit(e3nn_sph)
        jit_grad = jax.jit(jax.grad(single_scalar_output))
        jit_vmap_hessian = jax.jit(jax.vmap(jax.hessian(single_scalar_output), in_axes=(0,)))
        jit_jacfwd = jax.jit(jax.jacfwd(array_output))
        jit_jacrev = jax.jit(jax.jacrev(array_output))

    time_noderi = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = e3nn_sph(xyz)
        elapsed += time.time()
        time_noderi[i] = elapsed
    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" E3NN no derivatives: {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_noderi[:warmup])

    time_noderi[:] = 0.0
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        _ = jit_e3nn_sph(xyz)
        elapsed += time.time()
        time_noderi[i] = elapsed
    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" E3NN no der, jit:    {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_noderi[:warmup])

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        _ = jit_grad(xyz)
        elapsed += time.time()
        time_deri[i] = elapsed
    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" E3NN grad (jit):     {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        _ = jit_vmap_hessian(xyz)
        elapsed += time.time()
        time_deri[i] = elapsed
    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" E3NN hessian (jit):  {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_deri[:warmup])

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        _ = jit_jacfwd(xyz)
        elapsed += time.time()
        time_deri[i] = elapsed
    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" E3NN jacfwd (jit):   {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_deri[:warmup])

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        _ = jit_jacrev(xyz)
        elapsed += time.time()
        time_deri[i] = elapsed
    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" E3NN jacrev (jit):   {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose: print("Warm-up timings / sec.:\n", time_deri[:warmup])
    
    print(
        "******************************************************************************"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=200, help="number of samples")
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
            dtype=jnp.float64,
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
            dtype=jnp.float32,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )

    if args.gpu:
        raise ValueError("GPU implementation of sphericart-jax is not yet available")
