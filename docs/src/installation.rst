Installation
============

Only Unix-based systems are supported for the moment. We plan to support Windows and MacOS in the future.

Python package
--------------

The Python package can be installed with pip by simply running

.. code-block:: bash

    pip install sphericart

C library
---------

After cloning the repository with 

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart

.. code-block:: bash

    cd sphericart/src
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../lib/
    make install

This will install a shared object library inside the sphericart/lib/ folder.
See the examples section for more information on how to link the C shared 
object library in an external application.
