Installation
============


Python package
--------------

The Python package can be installed with pip by simply running

.. code-block:: bash

    pip install sphericart

C/C++ library
---------

After cloning the repository with 

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart

Default installation of the C/C++ library can be achieved as

.. code-block:: bash

    cd sphericart/
    mkdir build
    cd build/
    cmake ..
    make install

This will attempt install a static library inside the /usr/local/lib/ folder, 
which will almost certainly cause a permission error. The destination folder can
be changed by adding ``-DCMAKE_INSTALL_PREFIX=...`` to the ``cmake ..`` command.
In practice, we recommend ``cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local``, which
will be appropriate in the majority of cases.
