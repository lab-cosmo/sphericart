JAX API
===========

The `sphericart.jax` module aims to provide a functional-style and
`JAX`-friendly framework. As a result, it does not follow the same syntax as
the Python and PyTorch :py:class:`SphericalHarmonics` classes. Instead, it 
provides a function that is fully compatible with JAX primitives
(:py:class:`jax.grad`, :py:class:`jax.jit`, and so on).

Depending on the device the array is
stored on, as well as its `dtype`, the calculations will be performed
using 32- or 64- bits floating point arythmetics, and
using the CPU or CUDA implementation.

.. autofunction:: sphericart.jax.spherical_harmonics

.. autofunction:: sphericart.jax.solid_harmonics

