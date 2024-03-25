#!/usr/bin/env bash

set -eu

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

find . -type f \(                            \
        -name "*.c" -o -name "*.cpp"         \
        -o -name "*.h" -o -name "*.hpp"      \
        -o -name "*.cu" -o -name "*.cuh"     \
    \)                                       \
    -not -path "*/external/*"                \
    -not -path "*/build/*"                   \
    -not -path "*/.tox/*"                    \
    -exec clang-format -i {} \;
