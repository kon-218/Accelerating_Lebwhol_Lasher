from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy
import sys
import os

mpi_compile_args = os.popen("mpicc --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpicc --showme:link").read().strip().split(' ')

ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args = mpi_compile_args,
        extra_link_args    = mpi_link_args,
    )
]

setup(
    name='CythonVecLL',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()] 
)