#!/bin/bash

for file in $(find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -not -path "*/build/*" -not -path "*/.tox/*"); do
    clang-format -i "$file"
    if git diff --quiet "$file"; then
    echo "✅ $file is properly formatted."
    else
    echo "❌ $file is not properly formatted. Please run './scripts/format.sh' from the sphericart root directory"
    exit 1
    fi
done
