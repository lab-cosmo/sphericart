#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    exit 0
fi

cuda_major=$(
    nvidia-smi 2>/dev/null \
    | sed -nE 's/.*CUDA Version: *([0-9]+).*/\1/p' \
    | head -n 1
)

if [ -z "${cuda_major}" ] && command -v nvcc >/dev/null 2>&1; then
    cuda_major=$(
        nvcc --version 2>/dev/null \
        | sed -nE 's/.*release ([0-9]+).*/\1/p' \
        | head -n 1
    )
fi

if [ -z "${cuda_major}" ]; then
    echo "Could not detect CUDA version from nvidia-smi or nvcc" >&2
    exit 1
fi

case "${cuda_major}" in
    12|13)
        python -m pip install -U "cupy-cuda${cuda_major}x"
        ;;
    *)
        echo "Unsupported CUDA major version for CuPy wheels: ${cuda_major}" >&2
        exit 1
        ;;
esac
