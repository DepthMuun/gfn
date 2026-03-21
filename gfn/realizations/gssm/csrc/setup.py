import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Force sequential compilation to prevent NVCC Out-Of-Memory (LLVM ERROR)
os.environ["MAX_JOBS"] = "1"

# Directorio base
csrc_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(csrc_dir, "extension.cpp"),
    os.path.join(csrc_dir, "losses", "toroidal.cu"),
    os.path.join(csrc_dir, "geometry", "low_rank.cu"),
    os.path.join(csrc_dir, "integrators", "integrators.cpp")
]

# Configuración específica para MSVC/Windows vs Linux
extra_compile_args = {
    'cxx': ['-O2'],
    'nvcc': ['-O2', '-allow-unsupported-compiler']
}
if os.name == 'nt':
    extra_compile_args['cxx'].append('/std:c++17')

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            name='gfn_cuda',
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
