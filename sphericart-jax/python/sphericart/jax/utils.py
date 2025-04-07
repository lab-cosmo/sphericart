import jax


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


try:
    from .lib.sphericart_jax_cuda import build_sph_descriptor
except ImportError:

    def build_sph_descriptor(a, b):
        raise ValueError(
            "Trying to use sphericart-jax on CUDA, "
            "but sphericart-jax was installed without CUDA support. "
            "Please re-install sphericart-jax with CUDA support"
        )


class jax_float64:
    def __enter__(self):
        jax.config.update("jax_enable_x64", True)
        return self

    def __exit__(self, type, value, traceback):
        jax.config.update("jax_enable_x64", False)
