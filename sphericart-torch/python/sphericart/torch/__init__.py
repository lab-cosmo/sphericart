import glob
import os
import re
import sys
from collections import namedtuple

import torch

from .e3nn import e3nn_spherical_harmonics, patch_e3nn, unpatch_e3nn  # noqa: F401
from .spherical_hamonics import SolidHarmonics, SphericalHarmonics  # noqa: F401


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    torch_version = parse_version(torch.__version__)
    expected_prefix = os.path.join(
        _HERE, f"torch-{torch_version.major}.{torch_version.minor}"
    )
    if os.path.exists(expected_prefix):
        if sys.platform.startswith("darwin"):
            path = os.path.join(expected_prefix, "lib", "libsphericart_torch.dylib")
        elif sys.platform.startswith("linux"):
            path = os.path.join(expected_prefix, "lib", "libsphericart_torch.so")
        elif sys.platform.startswith("win"):
            path = os.path.join(expected_prefix, "bin", "sphericart_torch.dll")
        else:
            raise ImportError("Unknown platform. Please edit this file")

        if os.path.isfile(path):
            return path
        else:
            raise ImportError(
                "Could not find sphericart_torch shared library at " + path
            )

    # gather which torch version(s) the current install was built
    # with to create the error message
    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "../torch-*")):
        existing_versions.append(os.path.basename(prefix)[11:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load sphericart-torch with torch v{torch.__version__}, "
            f"but it was compiled against torch v{existing_versions[0]}, which "
            "is not ABI compatible"
        )
    else:
        all_versions = ", ".join(map(lambda version: f"v{version}", existing_versions))
        raise ImportError(
            f"Trying to load sphericart-torch with torch v{torch.__version__}, "
            f"we found builds for torch {all_versions}; which are not ABI compatible.\n"
            "You can try to re-install from source with "
            "`pip install sphericart-torch --no-binary=sphericart-torch`"
        )


# load the C++ operators
torch.ops.load_library(_lib_path())


def _sph_size(xyz, l_max):
    return (xyz.shape[0], (l_max + 1) ** 2)


def _xyz_grad_from_dsph(grad_sph, dsph):
    return torch.sum(grad_sph.unsqueeze(1) * dsph, dim=2)


def _xyz_grad_from_ddsph(grad_dsph, ddsph):
    return torch.sum(grad_dsph.unsqueeze(2) * ddsph, dim=(1, 3))


def _fake_compute(xyz, l_max):
    return xyz.new_empty(_sph_size(xyz, l_max))


def _fake_compute_with_gradients(xyz, l_max):
    sph = xyz.new_empty(_sph_size(xyz, l_max))
    dsph = xyz.new_empty((xyz.shape[0], 3, sph.shape[1]))
    return sph, dsph


def _fake_compute_with_hessians(xyz, l_max):
    sph, dsph = _fake_compute_with_gradients(xyz, l_max)
    ddsph = xyz.new_empty((xyz.shape[0], 3, 3, sph.shape[1]))
    return sph, dsph, ddsph


def _setup_fake_impls(prefix):
    @torch.library.register_fake(f"sphericart_torch::{prefix}")
    def _fake_compute_op(xyz, l_max, backward_second_derivatives=False):
        return _fake_compute(xyz, l_max)

    @torch.library.register_fake(f"sphericart_torch::{prefix}_with_gradients")
    def _fake_compute_with_gradients_op(xyz, l_max):
        return _fake_compute_with_gradients(xyz, l_max)

    @torch.library.register_fake(f"sphericart_torch::{prefix}_with_hessians")
    def _fake_compute_with_hessians_op(xyz, l_max):
        return _fake_compute_with_hessians(xyz, l_max)


def _setup_compute_context(ctx, inputs, output):
    xyz, l_max, backward_second_derivatives = inputs
    ctx.save_for_backward(xyz)
    ctx.l_max = l_max
    ctx.backward_second_derivatives = backward_second_derivatives


def _setup_gradient_context(ctx, inputs, output):
    xyz, l_max = inputs
    _, dsph = output
    ctx.save_for_backward(xyz, dsph)
    ctx.l_max = l_max


def _setup_autograd(prefix):
    gradients_op = getattr(torch.ops.sphericart_torch, f"{prefix}_with_gradients")
    hessians_op = getattr(torch.ops.sphericart_torch, f"{prefix}_with_hessians")

    def _compute_backward(ctx, grad_sph):
        (xyz,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None

        grad_context = (
            torch.enable_grad() if ctx.backward_second_derivatives else torch.no_grad()
        )
        with grad_context:
            _, dsph = gradients_op(xyz, ctx.l_max)

        return _xyz_grad_from_dsph(grad_sph, dsph), None, None

    def _gradients_backward(ctx, grad_sph, grad_dsph):
        xyz, dsph = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None

        xyz_grad = None
        if grad_sph is not None:
            xyz_grad = _xyz_grad_from_dsph(grad_sph, dsph)

        if grad_dsph is not None:
            with torch.no_grad():
                _, _, ddsph = hessians_op(xyz, ctx.l_max)
            ddsph_grad = _xyz_grad_from_ddsph(grad_dsph, ddsph)
            xyz_grad = ddsph_grad if xyz_grad is None else xyz_grad + ddsph_grad

        return xyz_grad, None

    torch.library.register_autograd(
        f"sphericart_torch::{prefix}",
        _compute_backward,
        setup_context=_setup_compute_context,
    )
    torch.library.register_autograd(
        f"sphericart_torch::{prefix}_with_gradients",
        _gradients_backward,
        setup_context=_setup_gradient_context,
    )


for _prefix in ("spherical_harmonics", "solid_harmonics"):
    _setup_fake_impls(_prefix)
    _setup_autograd(_prefix)
