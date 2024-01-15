import jax
from .lib import sphericart_jax

# register the operations to xla
for _name, _value in sphericart_jax.registrations().items():
    if _name.startswith("cpu_"):
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
    else:
        raise NotImplementedError(f"Unsupported target in sphericart_jax_cpu {_name}")

try:
    from .lib import sphericart_jax_gpu
    # register the operations to xla
    for _name, _value in sphericart_jax_gpu.registrations().items():
        if _name.startswith("cuda_"):
            jax.lib.xla_client.register_custom_call_target(_name, _value, platform="gpu")
        else:
            raise NotImplementedError(f"Unsupported target in sphericart_jax_gpu {_name}")
        
        
except:
    pass


    
