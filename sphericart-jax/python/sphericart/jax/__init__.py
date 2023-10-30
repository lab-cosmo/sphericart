import jax
from .lib import sphericart_jax_cpu
from .spherical_harmonics import spherical_harmonics


# register the CPU operation to xla
for _name, _value in sphericart_jax_cpu.registrations().items():
    jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
