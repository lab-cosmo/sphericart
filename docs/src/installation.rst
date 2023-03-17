Installation
============


Python package
--------------

The Python package can be installed with pip by simply running

.. code-block:: bash

    pip install sphericart

This basic package makes use of NumPy. A PyTorch-based implementation,
which includes GPU support, can be installed with 

.. code-block:: bash

    pip install sphericart?????????????????


C/C++ library
-------------

First, the repository needs to be cloned with 

.. code-block:: bash

    git clone https://github.com/lab-cosmo/sphericart

After that, default installation of the C/C++ library can be achieved as

.. code-block:: bash

    cd sphericart/
    mkdir build
    cd build/
    cmake ..
    make install

This will attempt to install a static library inside the /usr/local/lib/ folder, 
which will almost certainly cause a permission error. This can be solved by using 
administrator privileges (e.g., ``sudo make install``). Alternatively, the destination 
folder can be changed by adding ``-DCMAKE_INSTALL_PREFIX=...`` to the ``cmake ..`` command.
In practice, we recommend ``cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local``, which
will be appropriate in the vast majority of cases.
