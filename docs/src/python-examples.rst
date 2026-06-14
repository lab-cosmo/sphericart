Python
------
The ``Python`` API relies on a 
:py:class:`sphericart.SphericalHarmonics`  class
and is used in a similar way to the C++ classes, with the 
difference that, in line with pythonic mores, the 
``compute`` methods allocate and return their own arrays.

.. literalinclude:: ../../examples/python/example.py
    :language: python

The same calculators also accept CuPy arrays:

.. literalinclude:: ../../examples/python/cupy.py
    :language: python
