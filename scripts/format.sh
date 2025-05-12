#!/usr/bin/env bash

set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

which clang-format
clang-format --version

git ls-files '*.cpp' '*.c' '*.hpp' '*.h' '*.cu' '*.cuh' | xargs -L 1 clang-format -i
