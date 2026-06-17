import os
import sys

import sphericart.torch


def test_lib_path():
    path = sphericart.torch._lib_path()

    assert os.path.isfile(path)

    if sys.platform.startswith("darwin"):
        assert os.path.basename(path) == "libsphericart_torch.dylib"
    elif sys.platform.startswith("linux"):
        assert os.path.basename(path) == "libsphericart_torch.so"
    elif sys.platform.startswith("win"):
        assert os.path.basename(path) == "sphericart_torch.dll"
    else:
        raise AssertionError(f"unknown platform: {sys.platform}")
