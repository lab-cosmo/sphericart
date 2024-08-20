import os
import subprocess
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext


ROOT = os.path.realpath(os.path.dirname(__file__))
SPHERICART_ARCH_NATIVE = os.environ.get("SPHERICART_ARCH_NATIVE", "ON")

#
#
#
#
#


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "sphericart/jax")

        os.makedirs(build_dir, exist_ok=True)

        cmake_prefix_path = [pybind11.get_cmake_dir()]

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DSPHERICART_ARCH_NATIVE={SPHERICART_ARCH_NATIVE}",
            f"-DCMAKE_PREFIX_PATH={';'.join(cmake_prefix_path)}",
        ]

        CUDA_HOME = os.environ.get("CUDA_HOME")
        if CUDA_HOME is not None:
            cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}")
            cmake_options.append("-DSPHERICART_ENABLE_CUDA=ON")

        if sys.platform.startswith("darwin"):
            cmake_options.append("-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=11.0")

        # ARCHFLAGS is used by cibuildwheel to pass the requested arch to the
        # compilers
        ARCHFLAGS = os.environ.get("ARCHFLAGS")
        if ARCHFLAGS is not None:
            cmake_options.append(f"-DCMAKE_C_FLAGS={ARCHFLAGS}")
            cmake_options.append(f"-DCMAKE_CXX_FLAGS={ARCHFLAGS}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--parallel", "--target", "install"],
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
            "Use `pip install .` to install from source."
        )


if __name__ == "__main__":
    setup(
        version=open(os.path.join(ROOT, "sphericart", "VERSION")).readline().strip(),
        ext_modules=[
            Extension(name="sphericart_jax", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
        },
        package_data={
            "sphericart-jax": [
                "sphericart/jax/lib/*",
                "sphericart/jax/include/*",
            ]
        },
    )
