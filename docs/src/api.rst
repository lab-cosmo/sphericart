API documentation
=================

The core implementation of ``sphericart`` is written in C++. It relies on templates and
C++17 features such as ``if constexpr`` to reduce the runtime overhead of implementing
different normalization styles, and providing both a version with and without derivatives.

Spherical harmonics and their derivatives are computed with optimized hard-coded expressions
for low values of the principal angular momentum number :math:`l`, then switch to an efficient
recursive evaluation. The API involves initializing a calculator that allocates buffer space
and computes some constant factors, and then using it to compute :math:`Y_l^m` (and possibly its
first and/or second derivatives) for one or more points in 3D space.

This core C++ library is then made available to different environments through a C API.
This section contains a description of the interface of the ``sphericart`` library for the
different languages it supports.

.. toctree::
    :maxdepth: 1

    cpp-api
    c-api
    cuda-api
    python-api
    pytorch-api
    jax-api

Although the Julia API is not fully documented yet, basic usage examples are available
`here <https://github.com/lab-cosmo/sphericart/blob/main/julia/README.md>`_.
