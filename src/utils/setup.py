#!/usr/bin/env python

from setuptools import setup, Extension
import numpy

# Define the extension module
kernels_cpp_module = Extension(
    'kernels_cpp',
    sources=['kernels.cpp'],
    include_dirs=[
        numpy.get_include(),
        '/usr/include/python3.12',  # Python headers
        '/usr/include/python3.12/numpy',  # NumPy headers
    ],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', '-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='kernels_cpp',
    version='1.0',
    description='C++ implementation of SOAP kernels',
    ext_modules=[kernels_cpp_module],
    zip_safe=False,
)
