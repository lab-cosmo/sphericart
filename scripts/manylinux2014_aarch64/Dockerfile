# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux2014_aarch64

RUN apt-get update
RUN apt-key del 7fa2af80

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get install -y cuda-toolkit

# Configure dynamic linker run-time bindings
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Set environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda