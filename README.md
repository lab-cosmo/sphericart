# sphericart

This is sphericart, a multi-language library for the efficient calculation of the
spherical harmonics and their derivatives in Cartesian coordinates.

For instructions and examples on the usage of the library, please refer to our
[documentation](https://sphericart.readthedocs.io/en/latest/).


## Installation

### Python API

From source

```bash
git clone https://github.com/lab-cosmo/sphericart
pip install .

# if you also want the torch bindings
pip install ./torch
```

Pre-built (https://pypi.org/project/sphericart/), no torch bindings for now

```bash
pip sphericart
```

### C and C++ API

From source

```bash
git clone https://github.com/lab-cosmo/sphericart
cd shericart

mkdir build && cd build

cmake .. <cmake configuration options>
cmake --build . --target install
```

The following cmake configuration options are available:
- `-DSPHERICART_BUILD_TORCH=ON/OFF`: build the torch bindings in addition to the main library
- `-DSPHERICART_BUILD_TESTS=ON/OFF`: build C++ unit tests
- `-DSPHERICART_OPENMP=ON/OFF`: enable OpenMP parallelism
- `-DCMAKE_INSTALL_PREFIX=<where/you/want/to/install>` set the root path for installation
