# python setup.py build_ext --inplace
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob
import os

# specify src
_cwd = os.getcwd()
_ext_src_root = "./"
_ext_sources = \
    glob.glob("{}/*.cpp".format(_ext_src_root)) + \
    glob.glob("{}/*.cu".format(_ext_src_root))
_ext_include = os.path.join(_cwd, 'include')
# Standalone package
setup(
    name='fast_query',
    ext_modules=[
        CUDAExtension(
            name='_ext',
            sources=_ext_sources,
            extra_compile_args={
                'cxx': ['-O2', '-ffast-math', f'-I{_ext_include}'], 
                'nvcc': ['-O2', f'-I{_ext_include}']})
    ],
    cmdclass={ 
        'build_ext' : BuildExtension 
    }
)