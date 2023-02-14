Building and installing
=======================

To install the C library:

.. code-block:: bash

    cd src
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../lib/
    make install

To install the Python library:

.. code-block:: bash
    
    python -m pip install .
