import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel


ROOT = os.path.realpath(os.path.dirname(__file__))
SPHERICART_ARCH_NATIVE = os.environ.get("SPHERICART_ARCH_NATIVE", "ON")


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.7. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        import torch

        torch_major, torch_minor, *_ = torch.__version__.split(".")

        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(
            os.path.realpath(self.build_lib),
            f"sphericart/torch-{torch_major}.{torch_minor}",
        )

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DSPHERICART_TORCH_BUILD_FOR_PYTHON=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DSPHERICART_ARCH_NATIVE={SPHERICART_ARCH_NATIVE}",
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
        build_command = [
            "cmake",
            "--build",
            build_dir,
            "--parallel",
            "2",  # only two jobs to avoid OOM, we don't have many files
            "--target",
            "install",
        ]

        subprocess.run(build_command, check=True)


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

    try:
        import torch

        # if we have torch, we are building a wheel -  requires specific torch version
        torch_v_major, torch_v_minor, *_ = torch.__version__.split(".")
        torch_version = f"== {torch_v_major}.{torch_v_minor}.*"
    except ImportError:
        # otherwise we are building a sdist
        torch_version = ">= 2.1"

    install_requires = [f"torch {torch_version}"]

    setup(
        version=open(os.path.join(ROOT, "sphericart", "VERSION")).readline().strip(),
        install_requires=install_requires,
        ext_modules=[
            Extension(name="sphericart_torch", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
        package_data={
            "sphericart-torch": [
                "sphericart/torch/lib/*",
                "sphericart/torch/include/*",
            ],
        },
    )
