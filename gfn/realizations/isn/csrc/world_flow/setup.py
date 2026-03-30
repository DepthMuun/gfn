from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='gfn_world_flow',
    ext_modules=[
        CppExtension(
            name='gfn_world_flow',
            sources=['world_flow.cpp'],
            extra_compile_args=['-O3', '/O2', '/fp:fast'] if os.name == 'nt' else ['-O3', '-ffast-math']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)