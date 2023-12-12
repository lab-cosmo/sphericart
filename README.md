# sphericart

[![Test](https://github.com/lab-cosmo/sphericart/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lab-cosmo/sphericart/actions/workflows/tests.yml)

This is sphericart, a multi-language library for the efficient calculation of the
spherical harmonics and their derivatives in Cartesian coordinates.

For instructions and examples on the usage of the library, please refer to our
[documentation](https://sphericart.readthedocs.io/en/latest/).

<p align="center">
  <img src="./docs/src/_static/sphericart_3-0.png" alt="A plot of the +-1 isosurfaces of the Y^0\_3 solid harmonic, including also gradients." width="50%">
</p>

If you are using sphericart for your academic work, you can cite it as

```
@article{sphericart,
    title={Fast evaluation of spherical harmonics with sphericart},
    author={Bigi, Filippo and Fraux, Guillaume and Browning, Nicholas J. and Ceriotti, Michele},
    journal={J. Chem. Phys.},
    year={2023},
    number={159},
    pages={064802},
}
```

This library is dual-licensed under the Apache License 2.0 and the MIT license. You can use to use it under either of the two licenses.

## Installation

### Python

Pre-built (https://pypi.org/project/sphericart/).

```bash
pip install sphericart             # numpy version
pip install sphericart[torch]      # including also the torch bindings
pip install sphericart[jax]        # JAX bindings (CPU-only)
```

Note that the pre-built packages are compiled for a generic CPU, and might be
less performant than they could be on a specific processor. To generate
libraries that are optimized for the target system, you can build from source:

```bash
git clone https://github.com/lab-cosmo/sphericart
pip install .

# if you also want the torch bindings (CPU and GPU)
pip install .[torch]

# torch bindings, CPU-only version
pip install --extra-index-url https://download.pytorch.org/whl/cpu .[torch]
```

Building from source is also necessary to use sphericart's PyTorch GPU 
functionalities, and it requires a CUDA compiler.

### Julia 

A native Julia implementation of `sphericart` is provided, called `SpheriCart`. 
Install the package by opening a REPL, switch to the package manager by 
typing `]` and then `add SpheriCart`. 
See [julia/README.md](julia/README.md) for usage.


### C and C++

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
- `-DSPHERICART_BUILD_EXAMPLES=ON/OFF`: build C++ examples and benchmarks
- `-DSPHERICART_OPENMP=ON/OFF`: enable OpenMP parallelism
- `-DCMAKE_INSTALL_PREFIX=<where/you/want/to/install>` set the root path for installation


### Running tests and documentation

Tests and the local build of the documentation can be run with `tox`.
The default tests, which are also run on the CI, can be executed by simply running

```bash
tox
```

in the main folder of the repository.

To run tests in a CPU-only environment you can set the environment variable
`PIP_EXTRA_INDEX_URL` before calling tox, e.g.

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu tox -e docs
```

will build the documentation in a CPU-only environment.
