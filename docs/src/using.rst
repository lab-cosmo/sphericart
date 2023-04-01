Using the library
=================

`sphericart` is a multi-language library, providing similar interfaces across different 
programming languages, adapting to the typical usage patterns for the language. In general,
the calculators must be initialized first, to allocate buffers and initialize constant
factors that are used in the calculation. Then, spherical harmonics and their derivatives
can be computed for an array of 3D Cartesian points. 

All these examples perform the same task: computing Cartesian spherical harmonics for 
a random array of Cartesian coordinates, using both 32-bit and 64-bit floating points
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

? _ ?



