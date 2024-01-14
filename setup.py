import os
import subprocess
import sys
import uuid

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
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        source_dir = os.path.join(ROOT, "sphericart")
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "sphericart")

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_SHARED_LIBS=ON",
            f"-DSPHERICART_ARCH_NATIVE={SPHERICART_ARCH_NATIVE}",
        ]

        CUDA_HOME = os.environ.get("CUDA_HOME")
        if CUDA_HOME is not None:
            cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}")

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
            ["cmake", "--build", build_dir, "--target", "install"],
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
            + "install dist/sphericart-*.whl` to install from source."
        )


if __name__ == "__main__":
    SPHERICART_TORCH = os.path.realpath(os.path.join(ROOT, "sphericart-torch"))
    SPHERICART_JAX = os.path.realpath(os.path.join(ROOT, "sphericart-jax"))
    extras_require = {"torch": [], "jax": []}
    if os.path.exists(SPHERICART_TORCH):
        # we are building from a checkout

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for sphericart-torch, and force it to re-build from scratch
        uuid_ = uuid.uuid4()
        extras_require[
            "torch"
        ] = f"sphericart-torch @ file://{SPHERICART_TORCH}?{uuid_}"
    else:
        # installing wheel/sdist
        extras_require["torch"] = "sphericart-torch"

    if os.path.exists(SPHERICART_JAX):
        # we are building from a checkout

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for sphericart-jax, and force it to re-build from scratch
        uuid_ = uuid.uuid4()
        extras_require["jax"] = f"sphericart-jax @ file://{SPHERICART_JAX}?{uuid_}"
    else:
        # installing wheel/sdist
        extras_require["jax"] = "sphericart-jax"

    setup(
        version=open(os.path.join("sphericart", "VERSION")).readline().strip(),
        ext_modules=[
            Extension(name="sphericart", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
        package_data={
            "sphericart": [
                "sphericart/lib/*",
                "sphericart/include/*",
            ]
        },
        extras_require=extras_require,
    )
