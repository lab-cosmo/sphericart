Examples
========

We provide examples on how to calculate spherical harmonics from the C and Python
programming languages. We plan on supporting more languages in the future.

Python example
--------------

The Python example in ``examples/python/`` can be run with

.. code-block:: bash

    python example.py

This example calculates the spherical harmonics and their derivatives, and it 
benchmarks the evaluation times against the spherical harmonics implementation
of e3nn (https://github.com/e3nn/e3nn).



C example
---------

Once the installation of the C library has been completed (see the installation
section), a ``libsphericart.so`` shared library should have been created inside the
``lib/`` directory. In order to run the ``example.c`` program, the first step is to 
navigate to the ``examples/c/`` directory. Then, it is sufficient to run the 
following commands:

.. code-block:: bash

    make
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$PWD/../../lib/"
    ./example

The ``make`` command executes the ``Makefile`` in the directory. Note how this 
``Makefile`` includes the ``include/`` directory and adds ``lib/`` as a library
directory. The second command adds the adds ``lib/`` to the dynamic linking 
path.

Finally, the ``example`` executable benchmarks a few implementations of the 
spherical harmonics in C and it checks their internal consistency.
