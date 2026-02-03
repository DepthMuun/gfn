from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA source files
cuda_src_dir = os.path.join(os.path.dirname(__file__), 'src')

# List all CUDA source files
cuda_sources = [
    'cuda_kernels.cpp',
    # Geometry kernels
    'src/geometry/lowrank_christoffel.cu',
    'src/geometry/lowrank_christoffel_backward.cu',
    'src/geometry/lowrank_christoffel_friction_backward.cu',
    # Integrator kernels - Symplectic
    'src/integrators/symplectic/leapfrog_fused.cu',
    'src/integrators/symplectic/leapfrog_backward.cu',
    # Integrator kernels - Runge-Kutta
    'src/integrators/runge_kutta/heun_fused.cu',
    'src/integrators/runge_kutta/heun_backward.cu',
    'src/integrators/recurrent_manifold_fused.cu',
]

# Convert to absolute paths
cuda_sources = [os.path.join(os.path.dirname(__file__), src) for src in cuda_sources]

# Include directories
include_dirs = [
    cuda_src_dir,
    os.path.join(cuda_src_dir, 'common'),
    os.path.join(cuda_src_dir, 'geometry'),
    os.path.join(cuda_src_dir, 'integrators'),
]

is_windows = os.name == "nt"

cxx_flags = ["-O3"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_75,code=sm_75",
]

if is_windows:
    cxx_flags = ["/O2", "/bigobj", "/EHsc", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "/bigobj",
        "-Xcompiler",
        "/EHsc",
        "-Xcompiler",
        "/DNOMINMAX",
        "-Xcompiler",
        "/DWIN32_LEAN_AND_MEAN",
    ] + nvcc_flags[3:]

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            name='gfn_cuda',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
