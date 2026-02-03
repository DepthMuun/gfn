"""
Alternative ops.py for SCons-compiled kernels.
"""

import torch
import os
import ctypes

# Try to load precompiled SCons library
CUDA_AVAILABLE = False
gfn_cuda = None

try:
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(cuda_dir, 'gfn_cuda.pyd')
    
    if os.path.exists(lib_path):
        # Load via torch
        torch.ops.load_library(lib_path)
        gfn_cuda = torch.ops.gfn_cuda
        CUDA_AVAILABLE = True
        print(f"[GFN CUDA] Loaded precompiled kernels from {lib_path}")
    else:
        print(f"[GFN CUDA] Precompiled library not found: {lib_path}")
        print("[GFN CUDA] Run 'scons' to build, or fallback to JIT compilation")
        
        # Fallback to JIT compilation
        from torch.utils.cpp_extension import load
        gfn_cuda = load(
            name='gfn_cuda',
            sources=[
                os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                os.path.join(cuda_dir, 'kernels', 'christoffel_fused.cu'),
                os.path.join(cuda_dir, 'kernels', 'leapfrog_fused.cu'),
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False
        )
        CUDA_AVAILABLE = True
        print("[GFN CUDA] JIT compilation successful")
        
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"[GFN CUDA] Failed to load kernels: {e}")
    print("[GFN CUDA] Falling back to PyTorch implementation")


def christoffel_fused(v, U, W):
    """
    Fused Christoffel symbol computation: Γ(v,v) = W * (U^T v)^2
    """
    if CUDA_AVAILABLE and v.is_cuda:
        return gfn_cuda.christoffel_fused(v, U, W)
    else:
        # PyTorch fallback
        proj = torch.matmul(v, U)
        sq = proj * proj
        gamma = torch.matmul(sq, W.t())
        return torch.clamp(gamma, -5.0, 5.0)


def leapfrog_fused(x, v, f, U, W, dt, dt_scale=1.0):
    """
    Fused Leapfrog integration step.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.leapfrog_fused(x, v, f, U, W, dt, dt_scale)
    else:
        # PyTorch fallback
        effective_dt = dt * dt_scale
        
        gamma_v = christoffel_fused(v, U, W)
        v_half = v + 0.5 * effective_dt * (f - gamma_v)
        
        x_new = x + effective_dt * v_half
        
        gamma_v_half = christoffel_fused(v_half, U, W)
        v_new = v_half + 0.5 * effective_dt * (f - gamma_v_half)
        
        return x_new, v_new
