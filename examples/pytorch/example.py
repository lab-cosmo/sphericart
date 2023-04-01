import argparse

import numpy as np
import torch

import sphericart_torch

docstring = """
An example of the use of the PyTorch interface of the `sphericart` library.

Simply computes Cartesian spherical harmonics for the given parameters, for an
array of random 3D points, using both 32-bit and 64-bit arithmetics. 
"""


def sphericart_example(l_max=10, n_samples=10000, normalized=False):
    # `sphericart` provides a SphericalHarmonics object that initializes the
    # calculation and then can be called on any n x 3 arrays of Cartesian
    # coordinates. It computes _all_ SPH up to a given l_max, and can compute
    # scaled (default) and normalized (standard Ylm) harmonics.
    sh_calculator = sphericart_torch.SphericalHarmonics(l_max, normalized=normalized)

    # initializes the Cartesian coordinates of points
    xyz = torch.randn((n_samples, 3), dtype=torch.float64, device="cpu")

    # the interface allows to return directly the forward derivatives, 
    # similar to the Python version
    sh_sphericart, dsh_sphericart = sh_calculator.compute_with_gradients(xyz)

    xyz_f = xyz.clone().detach().type(torch.float32).to("cpu")
    sh_sphericart_f, dsh_sphericart_f = sh_calculator.compute_with_gradients(xyz_f)

    print(
        "Float vs double relative error: %12.8e\n"
        % (
            np.linalg.norm(sh_sphericart.detach() - sh_sphericart_f.detach())
            / np.linalg.norm(sh_sphericart.detach())
        )
    )

    # however, the implementation also supports backpropagation
    xyz = xyz.clone().detach().type(torch.float64).to("cpu").requires_grad_()
    sh_sphericart = sh_calculator.compute(xyz)

    sph_norm = torch.sum(sh_sphericart**2)
    sph_norm.backward()
    delta = torch.norm(xyz.grad- 2*torch.einsum("iaj,ij->ia",dsh_sphericart,sh_sphericart))
    print(f"Check derivative difference: {delta}" )


    


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
