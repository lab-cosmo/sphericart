Using the library
=================

`sphericart` is a multi-language library, providing similar interfaces across different 
programming languages, adapting to the typical usage patterns for the language. In general,
the calculators must be initialized first, to allocate buffers and initialize constant
factors that are used in the calculation. Then, spherical harmonics and their derivatives
can be computed for an array of 3D Cartesian points. 

All these examples perform the same task: computing Cartesian spherical harmonics for 
a random array of Cartesian coordinates, using either 32-bit or 64-bit floating-point
arithmetics, and compute the mean relative error. 

C++
---

.. literalinclude:: ../../examples/cpp/example.cpp
    :language: c++


C
---

.. literalinclude:: ../../examples/c/example.c
    :language: c

Python
------

.. literalinclude:: ../../examples/python/example.py
    :language: python


PyTorch
-------

The `PyTorch` implementation also supports backpropagation. 
The example shows how to compute gradients relative to the input
coordinates by using `backwards`. 
The :py:class:`sphericart.torch.SphericalHarmonics` object can also 
be used inside a :py:class:`torch.nn.Module`, that can then be 
compiled using `torchscript`. 

.. literalinclude:: ../../examples/pytorch/example.py
    :language: python


