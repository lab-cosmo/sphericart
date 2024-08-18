PyTorch API
===========

The main class for computing spherical harmonics using a
``torch``-compatible framework follows the same syntax as
the Python version :py:class:`sphericart.SphericalHarmonics`.
Depending on the ``device`` the tensor is
stored on, and its ``dtype``, the calculations will be performed
using 32- or 64- bits floating point arythmetics, and
using the CPU or CUDA implementation.

In short, although the :py:class:`sphericart.SphericalHarmonics`
class is technically not a ``torch.nn.Module``, it can be used in
the same way.

.. autoclass:: sphericart.torch.SphericalHarmonics
    :members:

The implementation also contains a couple of utility functions
to facilitate the integration of ``sphericart`` into code using
```e3nn``.

.. autofunction:: sphericart.torch.e3nn_spherical_harmonics

.. autofunction:: sphericart.torch.patch_e3nn

.. autofunction:: sphericart.torch.unpatch_e3nn
