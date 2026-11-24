Usage and examples
==================

`sphericart` is a multi-language library, providing similar interfaces across different 
programming languages and adapting to the typical usage patterns for the language. In general,
the calculators must be initialized first, to allocate buffers and initialize constant
factors that are used in the calculation. Then, spherical harmonics and their derivatives
can be computed for an array of 3D Cartesian points. 

This section provides a series of examples, which are also part of the source code repository
and are tested for consistency. In each language, the following examples compute Cartesian 
spherical harmonics for a random array of Cartesian coordinates, using either 32-bit or 64-bit 
floating-point arithmetics, and they evaluate the mean relative error between the two precision levels.

.. toctree::
    :maxdepth: 1

    cpp-examples
    c-examples
    cuda-examples
    python-examples
    pytorch-examples
    jax-examples
    spherical-complex

Although comprehensive Julia examples are not fully available yet, basic usage is illustrated
`here <https://github.com/lab-cosmo/sphericart/blob/main/julia/README.md>`_.
