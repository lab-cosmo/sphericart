import argparse

import cupy as cp
import numpy as np

import sphericart

docstring = """
An example of the CuPy interface of the `sphericart` library.

Computes Cartesian spherical harmonics on the GPU for a random array of 3D
points and compares the result against the NumPy CPU backend.
"""


def sphericart_cupy_example(l_max=10, n_samples=10000):
    xyz_cpu = np.random.rand(n_samples, 3)
    xyz_gpu = cp.asarray(xyz_cpu)

    sh_calculator = sphericart.SphericalHarmonics(l_max)

    sh_cpu = sh_calculator.compute(xyz_cpu)
    sh_gpu = sh_calculator.compute(xyz_gpu)

    print(
        "CPU vs GPU relative error: %12.8e"
        % (np.linalg.norm(sh_cpu - cp.asnumpy(sh_gpu)) / np.linalg.norm(sh_cpu))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=1000, help="number of samples")

    args = parser.parse_args()

    sphericart_cupy_example(args.l, args.s)
