import argparse

import numpy as np

import sphericart

docstring = """
An example of the use of the Python interface of the `sphericart` library.

Simply computes Cartesian spherical harmonics for the given parameters, for an
array of random 3D points, using both 32-bit and 64-bit arithmetics. 
"""


def sphericart_example(l_max=10, n_samples=10000, normalized=False):
    # `sphericart` provides a SphericalHarmonics object that initializes the
    # calculation and then can be called on any n x 3 arrays of Cartesian
    # coordinates. It computes _all_ SPH up to a given l_max, and can compute
    # scaled (default) and normalized (standard Ylm) harmonics.

    # ===== set up the calculation =====
    
    # initializes the Cartesian coordinates of points
    xyz = np.random.rand(n_samples, 3)

    # float32 version
    xyz_f = np.array(xyz, dtype=np.float32)
    
    # ===== API calls =====
    
    sh_calculator = sphericart.SphericalHarmonics(l_max, normalized=normalized)

    # without gradients
    sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=False)

    # with gradients
    sh_sphericart, dsh_sphericart = sh_calculator.compute(xyz, gradients=True)

    # the same calculator automatically chooses between float64 and float32
    sh_sphericart_f, dsh_sphericart_f = sh_calculator.compute(xyz_f, gradients=True)

    # ===== check results =====

    print(
        "Float vs double relative error: %12.8e"
        % (
            np.linalg.norm(sh_sphericart - sh_sphericart_f)
            / np.linalg.norm(sh_sphericart)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=1000, help="number of samples")
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="compute normalized spherical harmonics",
    )

    args = parser.parse_args()

    # Process everything.
    sphericart_example(args.l, args.s, args.normalized)
