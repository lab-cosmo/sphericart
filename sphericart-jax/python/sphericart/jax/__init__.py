import jax
from .lib import sphericart_jax_cpu


# register the operations to xla
for _name, _value in sphericart_jax_cpu.registrations().items():
    jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")

try:
    from .lib import sphericart_jax_cuda

    # register the operations to xla
    for _name, _value in sphericart_jax_cuda.registrations().items():
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="gpu")

except ImportError:
    pass
