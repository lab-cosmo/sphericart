#!/usr/bin/env python3
"""
This script updates the `Requires-Dist` information in sphericart-jax wheel METADATA
to contain the range of compatible jax versions. It expects newline separated
`Requires-Dist: jax ==...` information (corresponding to wheels built against a single
jax version) and will print `Requires-Dist: jax >=$MIN_VERSION,<${MAX_VERSION+1}` on
the standard output.

This output can the be used in the merged wheel containing the build against all jax
versions.
"""
import re
import sys


if __name__ == "__main__":
    jax_versions_raw = sys.argv[1]

    jax_versions = []
    for version in jax_versions_raw.split("\n"):
        if version.strip() == "":
            continue

        match = re.match(
            r"Requires-Dist: jax[ ]?==(\d+)\.(\d+)(?:\.(\d+))?(?:\.\*)?", version)
        if match is None:
            raise ValueError(f"unexpected Requires-Dist format: {version}")

        major, minor, patch, *_ = match.groups()
        major = int(major)
        minor = int(minor)
        patch = int(patch) if patch is not None else 0

        version = (major, minor, patch)

        if version in jax_versions:
            raise ValueError(f"duplicate jax version: {version}")

        jax_versions.append(version)

    jax_versions = list(sorted(jax_versions))

    min_version = f"{jax_versions[0][0]}.{jax_versions[0][1]}.{jax_versions[0][2]}"
    max_version = f"{jax_versions[-1][0]}.{jax_versions[-1][1]}.{jax_versions[-1][2] + 1}"

    print(f"Requires-Dist: jax >={min_version},<{max_version}")
