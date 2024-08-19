import argparse

import numpy as np
import torch

import sphericart.torch


docstring = """
An example of the use of the PyTorch interface of the `sphericart` library.

Simply computes Cartesian spherical harmonics for the given parameters, for an
array of random 3D points, using both 32-bit and 64-bit arithmetics.
"""


class SHModule(torch.nn.Module):
    """Example of how to use SphericalHarmonics from within a
    `torch.nn.Module`"""

    def __init__(self, l_max, normalized=False):
        if normalized:
            self.spherical_harmonics = sphericart.torch.SphericalHarmonics(l_max)
        else:
            self.spherical_harmonics = sphericart.torch.SolidHarmonics(l_max)
        super().__init__()

    def forward(self, xyz):
        sh = self.spherical_harmonics(xyz)  # or self.spherical_harmonics.compute(xyz)
        return sh


def sphericart_example(l_max=10, n_samples=10000, normalized=False):
    # `sphericart` provides a SphericalHarmonics object that initializes the
    # calculation and then can be called on any n x 3 arrays of Cartesian
    # coordinates. It computes _all_ SPH up to a given l_max, and can compute
    # scaled (default) and normalized (standard Ylm) harmonics.

    # ===== set up the calculation =====

    # initializes the Cartesian coordinates of points
    xyz = torch.randn((n_samples, 3), dtype=torch.float64, device="cpu")

    # float32 version
    xyz_f = xyz.clone().detach().type(torch.float32).to("cpu")

    # ===== API calls =====

    if normalized:
        sh_calculator = sphericart.torch.SphericalHarmonics(l_max)
    else:
        sh_calculator = sphericart.torch.SolidHarmonics(l_max)

    # the interface allows to return directly the forward derivatives (up to second
    # order), similar to the Python version
    sh_sphericart = sh_calculator.compute(xyz)  # same as sh_calculator(xyz)
    sh_sphericart, dsh_sphericart = sh_calculator.compute_with_gradients(xyz)
    (
        sh_sphericart,
        dsh_sphericart,
        ddsh_sphericart,
    ) = sh_calculator.compute_with_hessians(xyz)

    sh_sphericart_f, dsh_sphericart_f = sh_calculator.compute_with_gradients(xyz_f)

    # ===== check results =====

    print(
        "Float vs double relative error: %12.8e"
        % (
            np.linalg.norm(sh_sphericart.detach() - sh_sphericart_f.detach())
            / np.linalg.norm(sh_sphericart.detach())
        )
    )

    # ===== autograd integration =====

    # the implementation also supports back-propagation.
    # the input tensor must be tagged to have `requires_grad`
    xyz_ag = xyz.clone().detach().type(torch.float64).to("cpu").requires_grad_()
    sh_sphericart = sh_calculator.compute(xyz_ag)

    # then the spherical harmonics **but not their derivatives**
    # can be used with the usual PyTorch backward() workflow
    # nb: we sum only the even terms in the array because the total norm of a
    # Ylm is constant
    sph_norm = torch.sum(sh_sphericart[:, ::2] ** 2)
    sph_norm.backward()

    # checks the derivative is correct using the forward call
    delta = torch.norm(
        xyz_ag.grad
        - 2
        * torch.einsum("iaj,ij->ia", dsh_sphericart[:, :, ::2], sh_sphericart[:, ::2])
    ) / torch.norm(xyz_ag.grad)
    print(f"Check derivative difference (FW vs BW): {delta}")

    # double derivatives. In order to access them via back-propagation, an additional
    # flag must be specified at class instantiation:
    if normalized:
        sh_calculator_2 = sphericart.torch.SphericalHarmonics(
            l_max, backward_second_derivatives=True
        )
    else:
        sh_calculator_2 = sphericart.torch.SolidHarmonics(
            l_max, backward_second_derivatives=True
        )

    # double grad() call:
    xyz_ag2 = xyz[:5].clone().detach().type(torch.float64).to("cpu").requires_grad_()
    sh_sphericart_2 = sh_calculator_2.compute(xyz_ag2)
    sph_norm = torch.sum(sh_sphericart_2[:, ::2] ** 2)
    grad = torch.autograd.grad(sph_norm, xyz_ag2, retain_graph=True, create_graph=True)[
        0
    ]
    grad_grad = torch.autograd.grad(torch.sum(grad), xyz_ag2)[0]

    # hessian() call:
    xyz_ag2 = xyz[:5].clone().detach().type(torch.float64).to("cpu").requires_grad_()

    def func(xyz):
        sh_sphericart_2 = sh_calculator_2.compute(xyz)
        return torch.sum(sh_sphericart_2[:, ::2] ** 2)

    hessian = torch.autograd.functional.hessian(func, xyz_ag2)

    # ===== torchscript integration =====
    xyz_jit = xyz.clone().detach().type(torch.float64).to("cpu").requires_grad_()

    module = SHModule(l_max, normalized)

    # JIT compilation of the module
    script = torch.jit.script(module)
    sh_jit = script(xyz_jit)

    print(f"jit vs direct call: {torch.norm(sh_jit - sh_sphericart)}")

    # ===== GPU implementation ======

    if torch.cuda.is_available():
        xyz_cuda = xyz.clone().detach().type(torch.float64).to("cuda")

        sh_sphericart_cuda, dsh_sphericart_cuda = sh_calculator.compute_with_gradients(
            xyz_cuda
        )

        norm_dsph = torch.norm(dsh_sphericart_cuda.to("cpu") - dsh_sphericart)
        print(f"Check fw derivative difference CPU vs CUDA: {norm_dsph}")

        xyz_cuda_bw = (
            xyz.clone().detach().type(torch.float64).to("cuda").requires_grad_()
        )
        sh_sphericart_cuda_bw = sh_calculator.compute(xyz_cuda_bw)

        # then the spherical harmonics **but not their derivatives**
        # can be used with the usual PyTorch backward() workflow
        sph_norm_cuda = torch.sum(sh_sphericart_cuda_bw[:, ::2] ** 2)
        sph_norm_cuda.backward()

        delta = torch.norm(xyz_ag.grad - xyz_cuda_bw.grad.to("cpu")) / torch.norm(
            xyz_ag.grad
        )
        print(f"Check derivative difference CPU vs CUDA: {delta}")


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
