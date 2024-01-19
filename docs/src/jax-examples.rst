JAX
---

The ``jax`` implementation consists of the ``sphericart.jax.spherical_harmonics()``
function, which fits within idiomatic JAX code much better than the
corresponding ``numpy`` or ``torch`` classes.
The example shows how to compute spherical harmonics with the aforementioned
function, as well as how to transform it with standard JAX transformations
such as ``jax.vmap()``, ``jax.grad()``, ``jax.jit()`` and others.

.. literalinclude:: ../../examples/jax/example.py
    :language: python
