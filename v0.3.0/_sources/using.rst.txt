Usage and examples
==================

`sphericart` is a multi-language library, providing similar interfaces across different 
programming languages, adapting to the typical usage patterns for the language. In general,
the calculators must be initialized first, to allocate buffers and initialize constant
factors that are used in the calculation. Then, spherical harmonics and their derivatives
can be computed for an array of 3D Cartesian points. 

This section provides a series of examples, that are also part of the source code repository
and are tested for consistency, that perform the same task: computing Cartesian spherical harmonics for 
a random array of Cartesian coordinates, using either 32-bit or 64-bit floating-point
arithmetics, evaluating the mean relative error between the two precision levels. 

C++
---

The `SphericalHarmonics` class initializes and stores internally pre-factors and 
buffers, but the `compute` and `compute_with_gradients` functions require pre-allocated
`std::vector` to hold the results. See the :doc:`API documentation <api>` for alternative
calls using bare arrays or individual samples.

.. literalinclude:: ../../examples/cpp/example.cpp
    :language: c++


C
---
The C API uses an opaque type `sphericart_calculator_t` to hold a reference to the
C++ implementation of a `SphericalHarmonics` object. Several functions that mimic
the methods of the class can be called to compute spherical harmonics and their
derivatives. 

.. literalinclude:: ../../examples/c/example.c
    :language: c


Python
------
The `Python` API relies on a 
:py:class:`sphericart.SphericalHarmonics`  class
and is used in a similar way to the C++ classes, with the 
difference that, in line with pythonic mores, the 
`compute` methods allocate and return their own arrays.

.. literalinclude:: ../../examples/python/example.py
    :language: python


PyTorch
-------

The `PyTorch` implementation follows closely the syntax and usage of the 
`Python` implementation, but also supports backpropagation. 
The example shows how to compute gradients relative to the input
coordinates by using `backwards`. 
The :py:class:`sphericart.torch.SphericalHarmonics` object can also 
be used inside a :py:class:`torch.nn.Module`, that can then be 
compiled using `torchscript`. 

.. literalinclude:: ../../examples/pytorch/example.py
    :language: python


