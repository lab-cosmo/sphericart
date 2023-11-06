#!/bin/bash

find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -not -path "*/build/*" -not -path "*/.tox/*" | xargs clang-format -i
