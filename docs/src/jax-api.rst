JAX API
===========

The main class for computing spherical harmonics using a
`JAX`-compatible framework does not follow the same syntax as
the Python version :py:class:`sphericart.SphericalHarmonics` but provides an API
compatible with the main JAX primitives, e.g. :py:class:`jax.grad`, :py:class:`jax.jit`.


Depending on the device the array is
stored on, and its `dtype`, the calculations will be performed
using 32- or 64- bits floating point arythmetics, and
using the CPU implementation.

.. autoclass:: sphericart.jax.spherical_harmonics
    :members:

