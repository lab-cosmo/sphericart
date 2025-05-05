import jax


class jax_float64:
    def __enter__(self):
        jax.config.update("jax_enable_x64", True)
        return self

    def __exit__(self, type, value, traceback):
        jax.config.update("jax_enable_x64", False)
