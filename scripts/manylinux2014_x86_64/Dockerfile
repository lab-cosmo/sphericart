FROM quay.io/pypa/manylinux2014_x86_64

# Set environment variables for Python and CUDA versions
ARG PYTHON_VER="3.11"
ARG PYTHON_VER_NO_DOT="311"
ARG CUDA_VER="12.1"
ARG CUDA_VER_NO_DOT="121"
ARG CUDA_VER_DASH="12-1"
ARG PYTORCH_VERSION="2.4.1"

# Install system dependencies and CUDA
RUN yum install -y yum-utils gcc gcc-c++ make zlib-devel bzip2-devel libffi-devel \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && yum install -y \
    cuda-toolkit-${CUDA_VER_DASH} \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Set Python version environment variables dynamically
ENV PATH="/opt/python/cp${PYTHON_VER_NO_DOT}-cp${PYTHON_VER_NO_DOT}/bin:${PATH}"
RUN ln -sf /opt/python/cp${PYTHON_VER_NO_DOT}-cp${PYTHON_VER_NO_DOT}/bin/python${PYTHON_VER} /opt/python/cp${PYTHON_VER_NO_DOT}-cp${PYTHON_VER_NO_DOT}/bin/python


# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

# Verify the CUDA installation
RUN echo "CUDA_HOME: ${CUDA_HOME}" && \
    nvcc --version

RUN pip install setuptools wheel cmake pybind11 --upgrade
RUN pip install torch==${PYTORCH_VERSION}+cu${CUDA_VER_NO_DOT} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER_NO_DOT}
RUN pip install -U "jax[cuda12]"

RUN mkdir /workspace

# Default command (bash)
CMD ["/bin/bash"]
