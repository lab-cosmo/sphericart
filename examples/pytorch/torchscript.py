import argparse

import numpy as np
import torch

import sphericart.torch

import torch

docstring = """
An example of the use of TorchScript and the PyTorch intrface of the `sphericart` library.

Compiles the TorchScript-compatible module to compute Cartesian spherical harmonics 
for an array of random 3D points. 
"""

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

    xyz = torch.randn((args.s, 3), dtype=torch.float64, device="cpu", requires_grad=True)
    xyz_clone = xyz.detach().clone()
    xyz_clone.requires_grad=True

    sph_module = sphericart.torch.SphericalHarmonics(args.l, normalized=args.normalized)

    script = torch.jit.script(sph_module)

    output_script = script.forward(xyz)

    out_module, _ = sph_module.compute(xyz_clone)

    output_script.sum().backward()
    out_module.sum().backward()

    assert torch.equal(xyz.grad, xyz_clone.grad)
