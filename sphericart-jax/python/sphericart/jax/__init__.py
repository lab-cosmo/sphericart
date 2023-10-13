import jax
from .lib import cpu_ops
from .spherical_harmonics import spherical_harmonics


# register the CPU operation to xla
for _name, _value in cpu_ops.registrations().items():
    jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
