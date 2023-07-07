C
---
The C API uses an opaque type `sphericart_calculator_t` to hold a reference to the
C++ implementation of a `SphericalHarmonics` object. Several functions that mimic
the methods of the class can be called to compute spherical harmonics and their
derivatives. 

.. literalinclude:: ../../examples/c/example.c
    :language: c
