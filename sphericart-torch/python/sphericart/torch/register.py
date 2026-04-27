import glob
import os
import re
import sys
from collections import namedtuple

import torch


_REGISTERED = False
_HERE = os.path.realpath(os.path.dirname(__file__))
Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    raise ValueError("Invalid version string format")


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
        raise ImportError("Could not find sphericart_torch shared library at " + path)

    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "../torch-*")):
        existing_versions.append(os.path.basename(prefix)[11:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load sphericart-torch with torch v{torch.__version__}, "
            f"but it was compiled against torch v{existing_versions[0]}, which "
            "is not ABI compatible"
        )

    all_versions = ", ".join(map(lambda version: f"v{version}", existing_versions))
    raise ImportError(
        f"Trying to load sphericart-torch with torch v{torch.__version__}, "
        f"we found builds for torch {all_versions}; which are not ABI compatible.\n"
        "You can try to re-install from source with "
        "`pip install sphericart-torch --no-binary=sphericart-torch`"
    )


def _register_fake(op, outputs):
    @torch.library.register_fake(op)
    def fake(xyz, l_max):
        sph = xyz.new_empty((xyz.shape[0], (l_max + 1) ** 2))
        if outputs == 1:
            return sph
        dsph = xyz.new_empty((xyz.shape[0], 3, sph.shape[1]))
        if outputs == 2:
            return sph, dsph
        return sph, dsph, xyz.new_empty((xyz.shape[0], 3, 3, sph.shape[1]))


def _register_vmap(op, out_dims):
    def rule(info, in_dims, xyz, l_max):
        if in_dims[0] not in (None, 0):
            xyz = xyz.movedim(in_dims[0], 0)
        return op(xyz, l_max), out_dims

    torch.library.register_vmap(op, rule)


def _register_autograd(prefix):
    gradients = getattr(torch.ops.sphericart_torch, f"{prefix}_with_gradients").default
    hessians = getattr(torch.ops.sphericart_torch, f"{prefix}_with_hessians").default

    def setup_context(ctx, inputs, output):
        xyz, ctx.l_max = inputs
        saved = (xyz,) if isinstance(output, torch.Tensor) else (xyz, output[1])
        ctx.save_for_backward(*saved)

    def backward(ctx, *grad_outputs):
        if not ctx.needs_input_grad[0]:
            return None, None

        saved = ctx.saved_tensors
        xyz = saved[0]
        xyz_grad = None

        if grad_outputs[0] is not None:
            dsph = saved[1] if len(saved) == 2 else gradients(xyz, ctx.l_max)[1]
            xyz_grad = torch.sum(grad_outputs[0].unsqueeze(1) * dsph, dim=2)

        if len(grad_outputs) > 1 and grad_outputs[1] is not None:
            ddsph = hessians(xyz, ctx.l_max)[2]
            extra = torch.sum(grad_outputs[1].unsqueeze(2) * ddsph, dim=(1, 3))
            xyz_grad = extra if xyz_grad is None else xyz_grad + extra
            if not xyz.is_cuda:
                xyz_grad = xyz_grad.detach()

        return xyz_grad, None  # `None` is the placeholder gradient for `l_max`

    torch.library.register_autograd(
        f"sphericart_torch::{prefix}",
        backward,
        setup_context=setup_context,
    )
    torch.library.register_autograd(
        f"sphericart_torch::{prefix}_with_gradients",
        backward,
        setup_context=setup_context,
    )


def _register(prefix):
    # registers either spherical harmonics (prefix="spherical_harmonics")
    # or solid harmonics (prefix="solid_harmonics")
    value = getattr(torch.ops.sphericart_torch, prefix).default
    gradients = getattr(torch.ops.sphericart_torch, f"{prefix}_with_gradients").default
    hessians = getattr(torch.ops.sphericart_torch, f"{prefix}_with_hessians").default

    _register_fake(f"sphericart_torch::{prefix}", 1)
    _register_fake(f"sphericart_torch::{prefix}_with_gradients", 2)
    _register_fake(f"sphericart_torch::{prefix}_with_hessians", 3)
    _register_autograd(prefix)
    _register_vmap(value, 0)
    _register_vmap(gradients, (0, 0))
    _register_vmap(hessians, (0, 0, 0))


def register():
    global _REGISTERED
    if _REGISTERED:
        return

    torch.classes.load_library(_lib_path())
    _register("spherical_harmonics")
    _register("solid_harmonics")
    _REGISTERED = True
