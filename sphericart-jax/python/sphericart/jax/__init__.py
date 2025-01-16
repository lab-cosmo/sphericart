import jax
from packaging import version

from .lib import sphericart_jax_cpu
from .spherical_harmonics import spherical_harmonics, solid_harmonics  # noqa: F401


def get_minimum_cuda_version_for_jax(jax_version):
    """
    Get the minimum required CUDA version for a specific JAX version.

    Args:
        jax_version (str): Installed JAX version, e.g., '0.4.11'.

    Returns:
        tuple: Minimum required CUDA version as (major, minor), e.g., (11, 8).
    """
    # Define ranges of JAX versions and their corresponding minimum CUDA versions
    version_ranges = [
        (version.parse("0.4.26"), version.parse("999.999.999"),
         (12, 1)),  # JAX 0.4.26 and later: CUDA 12.1+
        (version.parse("0.4.11"), version.parse("0.4.25"),
         (11, 8)),        # JAX 0.4.11 - 0.4.25: CUDA 11.8+
    ]

    # Parse the current JAX version
    jax_ver = version.parse(jax_version)

    # Find the appropriate CUDA version range
    for start, end, cuda_version in version_ranges:
        if start <= jax_ver <= end:
            return cuda_version

    # Default to a safe version if no range matches
    raise ValueError(f"Unsupported JAX version: {jax_version}")


# register the operations to xla
for _name, _value in sphericart_jax_cpu.registrations().items():
    jax.lib.xla_client.register_custom_call_target(
        _name, _value, platform="cpu")

try:
    from .lib import sphericart_jax_cuda
    from .lib.sphericart_jax_cuda import get_cuda_runtime_version

    cuda_version = get_cuda_runtime_version()
    cuda_version = (cuda_version['major'], cuda_version['minor'])
    jax_version = jax.__version__
    required_version = get_minimum_cuda_version_for_jax(jax_version)
    if (cuda_version < required_version):
        raise RuntimeError(
            f"Incompatible setup detected:\n"
            f"- Installed CUDA version: {cuda_version[0]}.{cuda_version[1]}\n"
            f"- Installed JAX version: {jax_version}\n"
            f"- Minimum required CUDA version for JAX {jax_version}: {required_version[0]}.{required_version[1]}\n"
            f"Please upgrade your CUDA Toolkit to meet the requirements."
        )

    # register the operations to xla
    for _name, _value in sphericart_jax_cuda.registrations().items():
        jax.lib.xla_client.register_custom_call_target(
            _name, _value, platform="gpu")

except ImportError:
    pass
