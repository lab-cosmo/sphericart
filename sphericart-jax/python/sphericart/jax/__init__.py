import jax
from .lib import sphericart_jax
from .spherical_harmonics import spherical_harmonics


# register the operations to xla
for _name, _value in sphericart_jax.registrations().items():
    if _name.startswith("cpu_"):
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
    elif _name.startswith("cuda_"):
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="gpu")
    else:
        raise NotImplementedError(f"Unsupported target {_name}")
