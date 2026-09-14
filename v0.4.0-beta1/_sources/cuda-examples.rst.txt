CUDA C++
--------

The ``sphericart::cuda::SphericalHarmonics`` class automatically initializes and internally stores
pre-factors and buffers, and its usage is similar to the C++ API, although here the class provides
a single unified function for all purposes (values, gradients, and Hessians). This is
illustrated in the example below. The CUDA C++ API is undocumented at this time and subject
to change, but the example below should be sufficient to get started.

.. literalinclude:: ../../examples/cuda/example.cu
    :language: cuda
