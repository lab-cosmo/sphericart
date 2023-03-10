import argparse
import time

import numpy as np

import sphericart

docstring = """
An example of the use of the Python interface of the `sphericart` library.

Simply computes Cartesian spherical harmonics for the given parameters, for an
array of random 3D points. Also gets some timing information.
If the `e3nn` package is present, also compares results.
"""

try:
    import e3nn
    import torch

    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False


def sphericart_example(l_max=10, n_samples=10000, n_tries=100, normalized=False):
    xyz = np.random.rand(n_samples, 3)

    print(f"== Timings for computing spherical harmonics up to l={l_max} ==")

    # `sphericart` provides a SphericalHarmonics object that initializes the
    # calculation and then can be called on any n x 3 arrays of Cartesian
    # coordinates. It computes _all_ SPH up to a given l_max, and can compute
    # scaled (default) and normalized (standard Ylm) harmonics.
    sh_calculator = sphericart.SphericalHarmonics(l_max, normalized=normalized)
    start = time.time()
    for _ in range(n_tries):
        # if called with gradient = False, compute() returns a n x (l_max+1)**2
        # array holding all the spherical harmonics up to l_max, and None
        sh_sphericart, _ = sh_calculator.compute(xyz, gradients=False)
    finish = time.time()
    print(f"sphericart took {1e9*(finish-start)/n_tries/n_samples} ns/sample")

    if _HAS_E3NN:
        # e3nn expects [y,z,x]
        xyz_tensor = torch.tensor(xyz[:, [1, 2, 0]])
        sh = e3nn.o3.spherical_harmonics(
            list(range(l_max + 1)), xyz_tensor, normalize=normalized
        )  # allow compilation (??)
        start = time.time()
        for _ in range(n_tries):
            sh = e3nn.o3.spherical_harmonics(
                list(range(l_max + 1)), xyz_tensor, normalize=normalized
            )
        finish = time.time()
        assert np.allclose(sh, sh_sphericart)  # checks that the implementations match
        print(f"e3nn took {1e9*(finish-start)/n_tries/n_samples} ns/sample")

    print("== Timings including gradients for spherical harmonics ==")

    start = time.time()
    for _ in range(n_tries):
        # if called with gradient = True, compute() returns a n x 3 x(l_max+1)**2
        # array holding all the spherical harmonics derivatives up to l_max
        sh_sphericart, sh_derivatives = sh_calculator.compute(xyz, gradients=True)
        # also computes a dummy loss which is the sum of the sph values. useful
        # to compare with torch backprop
        dummy_loss = sh_sphericart.sum()
        loss_derivatives = sh_derivatives.sum(axis=2)
    finish = time.time()
    print(f"sphericart took {1e9*(finish-start)/n_tries/n_samples} ns/sample")

    if _HAS_E3NN:
        xyz_tensor.requires_grad = True

        # checks that the implementations match
        sh = e3nn.o3.spherical_harmonics(
            list(range(l_max + 1)), xyz_tensor, normalize=normalized
        )
        dummy_loss = torch.sum(sh)
        dummy_loss.backward()
        assert np.allclose(
            loss_derivatives[:, [1, 2, 0]], xyz_tensor.grad.detach().numpy()
        )

        # here we accumulate the gradient so it computes nonsense, but it should
        # be ok for timings
        start = time.time()
        for _ in range(n_tries):
            sh = e3nn.o3.spherical_harmonics(
                list(range(l_max + 1)), xyz_tensor, normalize=normalized
            )
            dummy_loss = torch.sum(sh)
            dummy_loss.backward()
        finish = time.time()

        print(f"e3nn took {1e9*(finish-start)/n_tries/n_samples} ns/sample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-t", type=int, default=100, help="number of iterations")
    parser.add_argument("-s", type=int, default=1000, help="number of samples")
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="compute normalized spherical harmonics",
    )

    args = parser.parse_args()

    # Process everything.
    sphericart_example(args.l, args.s, args.t, args.normalized)
