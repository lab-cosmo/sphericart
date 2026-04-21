Installation
============


Python package (including PyTorch and JAX)
------------------------------------------

The Python package can be installed with pip by simply running

.. code-block:: bash

    pip install sphericart

This basic package makes use of NumPy and CuPy. Implementations supporting PyTorch and JAX can be installed with

.. code-block:: bash

    pip install sphericart[torch]
    pip install sphericart[jax]

If you need an extra 5-10% of performance on CPU, you should build the code from source:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart
    pip install .

    # if you also want the torch bindings (CPU and GPU)
    pip install .[torch]
    # if you also want the jax bindings
    pip install .[jax]

    # torch bindings (CPU-only)
    pip install --extra-index-url https://download.pytorch.org/whl/cpu .[torch]

Before installing the JAX version of ``sphericart``, make sure you already have the JAX
library installed according to the `official JAX installation instructions
<https://jax.readthedocs.io/en/latest/installation.html>`_.

In addition, if you want to use the CUDA functionalities of sphericart (with CuPy,
torch or JAX), make sure you have installed the CUDA toolkit
and set up the environment variables ``CUDA_HOME``, ``LD_LIBRARY_FLAGS``, and ``PATH``
accordingly. In case you want to use CuPy, it should be installed separately.


Julia package
-------------

The native Julia package can be installed by opening a REPL,
switching to the package manager by typing ``]`` and then executing 
the command ``add SpheriCart``.


C/C++/CUDA library
------------------

First, you should clone the repository with

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart

After that, you can install the C/C++ library as

.. code-block:: bash

    cd sphericart/
    mkdir build
    cd build/
    cmake ..  # possibly include cmake configuration options here
    cmake --build . --target install

(A C++17 compiler is required.)

The following cmake configuration options are available:

- ``-DSPHERICART_BUILD_TORCH=ON/OFF``: build the torch bindings in addition to the main library (OFF by default)
- ``-DSPHERICART_BUILD_TESTS=ON/OFF``: build C++ unit tests (OFF by default)
- ``-DSPHERICART_BUILD_EXAMPLES=ON/OFF``: build C++ examples and benchmarks (OFF by default)
- ``-DSPHERICART_OPENMP=ON/OFF``: enable OpenMP parallelism (ON by default)
- ``-DSPHERICART_ENABLE_CUDA=ON/OFF``: build the CUDA backend (OFF by default)
- ``-DSPHERICART_ENABLE_SYCL=ON/OFF``: build the SYCL backend (OFF by default)
- ``-DSPHERICART_SYCL_DEVICE=all/cpu/gpu/accelerator``: select the device type used by the SYCL backend (``all`` by default)
- ``-DCMAKE_INSTALL_PREFIX=where/you/want/to/install``: set the root path for installation (``/usr/local`` by default)

Without specifying any options, the commands above will attempt to install 
a static library inside the ``/usr/local/lib/`` folder, which might cause a 
permission error. In that case you can change the destination folder. For example,
``cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local`` will be appropriate in the majority of cases.

Building the code on Windows is also supported, but it might require some manual configuration.
You can refer to the GitHub CI configuration files for examples of how to set up the build on Windows.
