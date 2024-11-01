# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux2014_aarch64

sudo apt-get update
sudo apt-key del 7fa2af80

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/aarch64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get install -y cuda-toolkit

# Configure dynamic linker run-time bindings
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Set environment variables
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDADIR=/usr/local/cuda