Spherical coordinates and/or complex harmonics
----------------------------------------------

The algorithms implemented in ``sphericart`` are designed to work with Cartesian
input positions and real spherical (or solid) harmonics. However, depending on the use 
case, it might be more convenient to harmonics from spherical coordinates and/or work
with complex harmonics.

Below, we provide a series of Python examples that illustrate how to use the
``sphericart`` library in such cases. The examples can be easily adapted to the
other languages supported by the library. All the examples are consistent with the
definitions of the spherical harmonics from Wikipedia.

The use of these (and similar) adaptors is not recommended for applications where
performance is critical, as they introduce some computational overhead.


Computing harmonics from spherical coordinates
**********************************************

This is a simple class that computes the spherical harmonics from spherical coordinates.

.. literalinclude:: ../../examples/python/spherical.py
    :language: python


Computing complex harmonics
***************************

This is a simple class that computes complex spherical harmonics.

.. literalinclude:: ../../examples/python/complex.py
    :language: python


Computing complex harmonics from spherical coordinates
******************************************************

This is a simple class that computes complex spherical harmonics from spherical coordinates.
Its correctness can be verified against the scipy implementation of the spherical harmonics.

.. literalinclude:: ../../examples/python/spherical_and_complex.py
    :language: python
