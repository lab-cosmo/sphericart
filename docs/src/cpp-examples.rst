C++
---

The ``SphericalHarmonics`` class automatically initializes and internally stores pre-factors and 
buffers, but the ``compute``, ``compute_with_gradients`` and ``compute_with_hessians`` functions
require the user to provide :literal:`std::vector`s to store the results. Since these :literal:`std::vector`s
are only resized if their shape is not appropriate, this allows the user to reutilize the
same memory across multiple callse, thereby avoiding unnecessary memory allocations. This is
illustrated in the example below. See the :ref:`C++ API <cpp-api>` for alternative calls
using bare arrays or individual samples.

.. literalinclude:: ../../examples/cpp/example.cpp
    :language: c++
