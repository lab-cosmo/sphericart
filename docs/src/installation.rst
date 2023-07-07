Installation
============


Python package
--------------

The Python package can be installed with pip by simply running

.. code-block:: bash

    pip install sphericart

This basic package makes use of NumPy. A PyTorch-based implementation can be installed with

.. code-block:: bash

    pip install sphericart[torch]

This pre-built version available on PyPI sacrifices some performance to ensure it
can run on all systems, and it does not include GPU support.
If you need an extra 5-10% of performance or you want to evaluate the spherical harmonics on GPUs,
you should build the code from source:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart
    pip install .

    # if you also want the torch bindings (CPU and GPU)
    pip install .[torch]

    # torch bindings (CPU-only)
    pip install --extra-index-url https://download.pytorch.org/whl/cpu .[torch]



C/C++ library
-------------

First, you should clone the repository with

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart

After that, you can install the C/C++ library as

.. code-block:: bash

    cd sphericart/
    mkdir build
    cd build/
    cmake ..  # possibly include cmake configuration options here
    make install

(A C++17 compiler is required.)

The following cmake configuration options are available:

- ``-DSPHERICART_BUILD_TORCH=ON/OFF``: build the torch bindings in addition to the main library (OFF by default)
- ``-DSPHERICART_BUILD_TESTS=ON/OFF``: build C++ unit tests (OFF by default)
- ``-DSPHERICART_BUILD_EXAMPLES=ON/OFF``: build C++ examples and benchmarks (OFF by default)
- ``-DSPHERICART_OPENMP=ON/OFF``: enable OpenMP parallelism (ON by default)
- ``-DCMAKE_INSTALL_PREFIX=where/you/want/to/install``: set the root path for installation (``/usr/local`` by default)

Without specifying any options, the commands above will attempt to install 
a static library inside the ``/usr/local/lib/`` folder, which might cause a 
permission error. In that case you can change the destination folder. For example,
``cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local`` will be appropriate in the majority of cases.
