from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='gfn_topology',
    ext_modules=[
        CUDAExtension(
            name='gfn_topology',
            sources=[
                'topology.cpp',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
