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
pip install .[torch]

# torch bindings, CPU-only version
pip install --extra-index-url https://download.pytorch.org/whl/cpu .[torch]
```

Pre-built (https://pypi.org/project/sphericart/).

```bash
pip install sphericart             # numpy version
pip install sphericart[torch]      # including also the torch bindings
```

### C and C++ API

From source

```bash
git clone https://github.com/lab-cosmo/sphericart
cd sphericart

mkdir build && cd build

cmake .. <cmake configuration options>
cmake --build . --target install
```

The following cmake configuration options are available:
- `-DSPHERICART_BUILD_TORCH=ON/OFF`: build the torch bindings in addition to the main library
- `-DSPHERICART_BUILD_TESTS=ON/OFF`: build C++ unit tests
- `-DSPHERICART_OPENMP=ON/OFF`: enable OpenMP parallelism
- `-DCMAKE_INSTALL_PREFIX=<where/you/want/to/install>` set the root path for installation


### Running tests and documentation

Tests and local documentations can be run with `tox`. 
To run tests in a CPU-only environment you can set the environment variable
`PIP_EXTRA_INDEX_URL` before calling tox, e.g. 

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu tox -e docs
```

will build the documentation in a CPU-only environment. 
