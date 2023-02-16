import os
import subprocess
from distutils.command.build_ext import build_ext
from distutils.command.install import install as distutils_install
from setuptools import Extension, setup, find_packages


here = os.path.realpath(os.path.dirname(__file__))

class cmake_ext(build_ext):

    def run(self):
        source_dir = os.path.join(here, "src")
        build_dir = os.path.join(here, "build", "cmake")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "sphericart", "lib")
        
        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        print()
        print("INSTALL DIR", install_dir)
        print()

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}"
        ]

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install"],
            check=True,
        )

with open('README.md') as f:
    long_description = f.read()

setup(
    name="sphericart",
    version="0.0.1",
    description="sphericart, a package to calculate spherical harmonics efficiently in Cartesian coordinates",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=[
        Extension(name="sphericart", sources=[]),
    ],
    cmdclass={
        "build_ext": cmake_ext,
        "install": distutils_install
    }
)
