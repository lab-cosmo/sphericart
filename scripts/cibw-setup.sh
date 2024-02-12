#!bin/bash

echo "Running pre-build commands"
pip install wheel
pip install cmake
pip install setuptools
echo "installing torch+cu$1"
pip install torch --index-url https://download.pytorch.org/whl/cu$1
python -c "import torch; print (torch.__version__);"