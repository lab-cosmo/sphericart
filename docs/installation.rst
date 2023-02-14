# Building and installing

To install the C library:

```
cd src
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../lib/
make install
```

To install the Python library:

```python -m pip install .```
