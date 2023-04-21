import argparse

import numpy as np
import torch

import sphericart.torch

import torch

docstring = """
An example of the use of TorchScript and the PyTorch intrface of the `sphericart` library.

Creates a TorchScript-compatible module to compute Cartesian spherical harmonics 
for an array of random 3D points. 
"""


class SphericalHarmonicsModule(torch.nn.Module):
    def __init__(self, lmax, normalized):
        super().__init__()
        self.sh_calculator = sphericart.torch.SphericalHarmonics(
            lmax, normalized=normalized
        )

    def forward(self, xyz):
        sph, _ = self.sh_calculator.compute(xyz)

        return sph


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

    xyz = torch.randn((args.s, 3), dtype=torch.float64, device="cpu")

    script = torch.jit.script(SphericalHarmonicsModule(args.l, args.normalized))

    output = script.forward(xyz)
