from setuptools import setup, find_packages
from torch.utils import cpp_extension

cuda_spherical_harmonics_extension = cpp_extension.CUDAExtension("spherical_harmonics_cuda_extension", 
            ["spherical_harmonics.cu"
        ],
         extra_compile_args={'cxx': ['-O2'],
                            'nvcc': ['-O2']})
    

ext_modules = [cuda_spherical_harmonics_extension]

setup(name="sph_cuda",
      packages = find_packages(),
      ext_modules = ext_modules,
      cmdclass={"build_ext": cpp_extension.BuildExtension})
