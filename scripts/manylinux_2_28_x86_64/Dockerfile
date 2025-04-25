# Inspired by https://github.com/ameli/manylinux-cuda, adding cuda-nvtx

FROM quay.io/pypa/manylinux_2_28_x86_64

# ------------
# Install cuda
# ------------

ARG CUDA_VER="12-6"
ARG ARCH="x86_64"

RUN yum install -y yum-utils
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
RUN yum -y install cuda-compiler-${CUDA_VER}.${ARCH} \
                   cuda-libraries-${CUDA_VER}.${ARCH} \
                   cuda-libraries-devel-${CUDA_VER}.${ARCH} \
                   cuda-nvtx-${CUDA_VER}.${ARCH}
RUN yum clean all
RUN rm -rf /var/cache/yum/*
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# -------------------------
# Set environment variables
# -------------------------

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

# --------
# Commands
# --------

CMD ["/bin/bash"]
