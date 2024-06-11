#!/usr/bin/env bash

# This script removes all temporary files created by Python during
# installation and tests running.

set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

rm -rf dist
rm -rf build

rm -rf sphericart-torch/dist
rm -rf sphericart-torch/build

rm -rf sphericart-jax/dist
rm -rf sphericart-jax/build

find . -name "*.egg-info" -exec rm -rf "{}" +
find . -name "__pycache__" -exec rm -rf "{}" +
find . -name ".coverage" -exec rm -rf "{}" +
