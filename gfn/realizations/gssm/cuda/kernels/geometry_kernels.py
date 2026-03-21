"""
Geometry Kernels — GFN V5
Unified entry points for geometric computations with hardware dispatching.
"""

import torch
from typing import Optional, Tuple, Union, Any
from ...registry import GEOMETRY_REGISTRY
from ...cuda import is_cuda_active

# Lazy import for CUDA ops to avoid loading if not available
_christoffel_cuda = None

def _get_cuda_ops():
    global _christoffel_cuda
    if _christoffel_cuda is None:
        try:
            from ...cuda.ops import christoffel_cuda_fwd
            _christoffel_cuda = christoffel_cuda_fwd
        except ImportError:
            pass
    return _christoffel_cuda

def unified_christoffel_fwd(
    x: torch.Tensor, 
    v: torch.Tensor, 
    U: torch.Tensor, 
    W: torch.Tensor, 
    clamp_val: float = 5.0,
    **kwargs: Any
) -> torch.Tensor:
    """
    Unified forward pass for Christoffel symbols.
    Dispatches to CUDA kernel if available and on GPU, otherwise falls back to PyTorch.
    """
    if is_cuda_active(v):
        cuda_op = _get_cuda_ops()
        if cuda_op is not None:
            try:
                return _run_cuda_christoffel(x, v, U, W, clamp_val, cuda_op, **kwargs)
            except Exception as e:
                # print(f"[Dispatcher] CUDA Error: {e}. Falling back.")
                pass
    
    return _run_pytorch_christoffel(x, v, U, W, clamp_val, **kwargs)

def _run_cuda_christoffel(x, v, U, W, clamp_val, cuda_op, **kwargs):
    # Kernel expects Head-Aware tensors [B, H, HD]
    # Check if x,v are [B, D] or [B, H, HD]
    if v.dim() == 2:
        x_k, v_k = x.unsqueeze(1), v.unsqueeze(1)
    else:
        x_k, v_k = x, v
        
    # U, W handling (assuming LowRank format)
    # This logic matches the legacy kernel expectations
    # W_k handling (ensuring rank-R is preserved per output dimension)
    if U.dim() == 3:
        U_k = U.transpose(1, 2).contiguous() # [H, R, HD]
        W_k = W.transpose(1, 2).contiguous() # [H, R, HD]
    else:
        U_k = U.T.unsqueeze(0).contiguous() # [1, R, HD]
        W_k = W.T.unsqueeze(0).contiguous() # [1, R, HD]

    # Execute CUDA kernel
    gamma = cuda_op(U_k, W_k, x_k, v_k, 0, 2.0, 1.0, 0.0) 
    
    if v.dim() == 2:
        gamma = gamma.squeeze(1)
    
    return clamp_val * torch.tanh(gamma / clamp_val)

def _run_pytorch_christoffel(x, v, U, W, clamp_val, **kwargs):
    # Multi-head PyTorch fallback
    if v.dim() == 3:
        B, H, HD = v.shape
        # Flatten batch and heads to use efficient matmuls
        v_flat = v.reshape(B * H, HD)
        if U.dim() == 3:
            # U: [H, HD, R], W: [H, R, HD]
            # Need to apply per-head
            proj = torch.bmm(v.transpose(0, 1), U).transpose(0, 1) # [B, H, R]
            sq = proj * proj
            W_t = W.transpose(-1, -2)
            gamma = torch.bmm(sq.transpose(0, 1), W_t).transpose(0, 1) # [B, H, HD]
        else:
            # Shared U, W across heads
            proj = torch.matmul(v_flat, U) # [B*H, R]
            sq = proj * proj
            gamma_flat = torch.matmul(sq, W.t()) # [B*H, HD]
            gamma = gamma_flat.view(B, H, HD)
    else:
        # Single head [B, D]
        proj = torch.matmul(v, U[0] if U.dim() == 3 else U)
        sq = proj * proj
        W_t = (W[0] if W.dim() == 3 else W).t()
        gamma = torch.matmul(sq, W_t)
    
    return clamp_val * torch.tanh(gamma / clamp_val)
