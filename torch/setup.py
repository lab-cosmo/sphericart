import os
import subprocess
import sys

import cmake
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext

ROOT = os.path.realpath(os.path.dirname(__file__))


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "sphericart_torch")

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DSPHERICART_TORCH_BUILD_FOR_PYTHON=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # "-DCMAKE_FIND_DEBUG_MODE=ON",
        ]

        CMAKE_EXE = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
        subprocess.run(
            [CMAKE_EXE, source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            [CMAKE_EXE, "--build", build_dir, "--target", "install"],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "install dist/equistore-*.whl` to install from source."
        )


if __name__ == "__main__":
    setup(
        version=open(os.path.join("sphericart", "VERSION")).readline().strip(),
        ext_modules=[
            Extension(name="sphericart_torch", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
        },
        package_data={
            "sphericart": [
                "sphericart/lib/*",
                "sphericart/include/*",
            ]
        },
    )
