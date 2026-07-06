"""
Geometry Kernels — GFN V5 (Improved)
Unified entry points for geometric computations with hardware dispatching.

Changes vs original:
  - Fixed tensor layout: removed erroneous transpose(1,2) that mismatched C++ kernel.
    C++ low_rank_christoffel_fwd expects v:[B,H,D], U:[H,D,R], W:[H,D,R] directly.
  - Added enable_trace_norm and is_paper_version kwargs pass-through.
  - Added toroidal_wrap helper that dispatches to the new standalone CUDA kernel.
  - Fallback PyTorch path preserved and corrected for the fixed layout.
"""

import torch
from typing import Optional, Any
from ...cuda import is_cuda_active

# Lazy import
_low_rank_fwd = None
_toroidal_wrap = None

def _get_cuda_ops():
    global _low_rank_fwd, _toroidal_wrap
    if _low_rank_fwd is None:
        try:
            from ...cuda.ops import low_rank_christoffel_fwd, toroidal_wrap_fwd
            _low_rank_fwd  = low_rank_christoffel_fwd
            _toroidal_wrap = toroidal_wrap_fwd
        except (ImportError, AttributeError):
            pass
    return _low_rank_fwd, _toroidal_wrap


def unified_christoffel_fwd(
    x: torch.Tensor,
    v: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    clamp_val: float = 5.0,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Unified forward pass for Christoffel symbols.

    Expected tensor layouts (same as C++ kernel):
      v : [B, H, D]  or  [B, D]
      U : [H, D, R]  or  [D, R]
      W : [H, D, R]  or  [D, R]

    Dispatches to the compiled CUDA kernel when available, otherwise
    falls back to pure PyTorch.
    """
    cuda_fwd, _ = _get_cuda_ops()

    if is_cuda_active(v) and cuda_fwd is not None:
        try:
            enable_trace_norm = bool(kwargs.get("enable_trace_norm", True))
            is_paper_version  = bool(kwargs.get("is_paper_version",  False))
            return _run_cuda_christoffel(v, U, W, clamp_val,
                                         enable_trace_norm, is_paper_version,
                                         cuda_fwd)
        except Exception:
            pass  # fall through to PyTorch

    return _run_pytorch_christoffel(v, U, W, clamp_val, **kwargs)


def _run_cuda_christoffel(
    v: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    clamp_val: float,
    enable_trace_norm: bool,
    is_paper_version: bool,
    cuda_op,
) -> torch.Tensor:
    """
    Calls low_rank_christoffel_fwd with the correct [B, H, D] + [H, D, R] layout.
    The C++ kernel does NOT want an extra transpose — removed the erroneous one.
    """
    # Ensure 3-D batch dim: [B, D] → [B, 1, D]
    squeeze = v.dim() == 2
    v_k = v.unsqueeze(1).contiguous() if squeeze else v.contiguous()

    # Ensure head dim in U, W: [D, R] → [1, D, R]
    U_k = U.unsqueeze(0).contiguous() if U.dim() == 2 else U.contiguous()
    W_k = W.unsqueeze(0).contiguous() if W.dim() == 2 else W.contiguous()

    gamma = cuda_op(v_k, U_k, W_k, clamp_val, enable_trace_norm, is_paper_version)

    return gamma.squeeze(1) if squeeze else gamma


def _run_pytorch_christoffel(
    v: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    clamp_val: float,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch fallback, corrected for [B, H, D] / [H, D, R] layout."""
    enable_trace_norm = bool(kwargs.get("enable_trace_norm", True))
    is_paper_version  = bool(kwargs.get("is_paper_version",  False))

    if v.dim() == 3:
        # [B, H, D] × [H, D, R] → [B, H, R]
        B, H, D = v.shape
        if U.dim() == 3:
            v_h  = v.permute(1, 0, 2)              # [H, B, D]
            vr_h = torch.bmm(v_h, U)               # [H, B, R]
            vr   = vr_h.permute(1, 0, 2)           # [B, H, R]
        else:
            # Shared U across heads
            vr = torch.matmul(v.reshape(B * H, D), U).view(B, H, -1)

        sq = vr.pow(2)
        if is_paper_version:
            sq = sq / (1.0 + torch.norm(vr, 2, -1, True))

        if W.dim() == 3:
            sq_h    = sq.permute(1, 0, 2)          # [H, B, R]
            gamma_h = torch.bmm(sq_h, W.transpose(-1, -2))  # [H, B, D]
            gamma   = gamma_h.permute(1, 0, 2)     # [B, H, D]
        else:
            gamma = torch.matmul(sq.reshape(B * H, -1), W.t()).view(B, H, D)

    else:
        # Single head [B, D]
        U_ = U[0] if U.dim() == 3 else U
        W_ = W[0] if W.dim() == 3 else W
        vr    = torch.matmul(v, U_)
        sq    = vr.pow(2)
        if is_paper_version:
            sq = sq / (1.0 + torch.norm(vr, 2, -1, True))
        gamma = torch.matmul(sq, W_.t())

    if enable_trace_norm:
        gamma = gamma - gamma.mean(-1, keepdim=True)

    return clamp_val * torch.tanh(gamma / clamp_val)


def unified_toroidal_wrap(x: torch.Tensor) -> torch.Tensor:
    """
    Fast toroidal wrap: x → [-π, π)
    Uses the standalone CUDA kernel when available (float4 vectorised),
    otherwise falls back to torch.remainder.
    """
    _, cuda_wrap = _get_cuda_ops()
    if is_cuda_active(x) and cuda_wrap is not None:
        try:
            return cuda_wrap(x.contiguous())
        except Exception:
            pass
    return torch.remainder(x + 3.141592653589793, 6.283185307179586) - 3.141592653589793
