# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux2014_x86_64

# ------------
# Install cuda
# ------------

ARG VER="12-4"
ARG ARCH="x86_64"

RUN yum install -y yum-utils
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
RUN yum -y install cuda-compiler-${VER}.${ARCH} \
                    cuda-libraries-${VER}.${ARCH} \
                    cuda-libraries-devel-${VER}.${ARCH}
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