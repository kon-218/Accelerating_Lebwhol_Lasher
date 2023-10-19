from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy
import sys

openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    name='CythonLebwohlLasher',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()] 
)