PyTorch API
===========

The classes for computing spherical harmonics using a
``torch``-compatible framework follow the same syntax as
the Python versions :py:class:`sphericart.SphericalHarmonics`
and :py:class:`sphericart.SolidHarmonics`, while inheriting
from ``torch.nn.Module``.
Depending on the ``device`` the tensor is
stored on, and its ``dtype``, the calculations will be performed
using 32- or 64- bits floating point arythmetics, and
using the CPU or CUDA implementation.

.. autoclass:: sphericart.torch.SphericalHarmonics
    :members:

.. autoclass:: sphericart.torch.SolidHarmonics
    :members:

The implementation also contains a couple of utility functions
to facilitate the integration of ``sphericart`` into code using
```e3nn``.

.. autofunction:: sphericart.torch.e3nn_spherical_harmonics

.. autofunction:: sphericart.torch.patch_e3nn

.. autofunction:: sphericart.torch.unpatch_e3nn
