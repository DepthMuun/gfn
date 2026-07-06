"""
Integrator Kernels — GFN V5 (Improved)
Unified entry points for numerical integration with hardware dispatching.

Changes vs original:
  - Fixed W_k reshape: was incorrectly mean-pooling over R dimension (information loss).
    Now passes full [H, D, R] tensor matching the C++ leapfrog_fwd signature.
  - Removed Python `for _ in range(steps)` loop — C++ kernel already loops internally.
  - Added DEBUG_SYNC environment flag for explicit synchronisation during profiling.
"""

import os
import torch
from typing import Optional, Tuple, Any
from ...cuda import is_cuda_active

# Set DEBUG_SYNC=1 to add explicit cuda.synchronize() for profiling
_DEBUG_SYNC = os.environ.get("GFN_DEBUG_SYNC", "0") == "1"

# Lazy imports for CUDA kernels
_leapfrog_fused = None
_yoshida_fused  = None

def _get_cuda_integrators():
    global _leapfrog_fused, _yoshida_fused
    if _leapfrog_fused is None:
        try:
            from ...cuda.ops import leapfrog_fused, yoshida_fused
            _leapfrog_fused = leapfrog_fused
            _yoshida_fused  = yoshida_fused
        except ImportError:
            pass
    return _leapfrog_fused, _yoshida_fused


def unified_leapfrog_step(
    x: torch.Tensor,
    v: torch.Tensor,
    force: Optional[torch.Tensor],
    U: torch.Tensor,
    W: torch.Tensor,
    dt: float,
    steps: int = 1,
    **kwargs
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified Leapfrog integration step.

    Dispatches to the compiled C++ / CUDA kernel (leapfrog_fwd) when available.
    The kernel already loops `steps` internally — no Python-level loop needed.
    Returns (None, None) to signal fallback if the kernel is unavailable.
    """
    if not is_cuda_active(v):
        return None, None

    f_leapfrog, _ = _get_cuda_integrators()
    if f_leapfrog is None:
        return None, None

    try:
        # ── Prepare tensors ────────────────────────────────────────────────
        # Ensure contiguous layout expected by C++ kernel
        x_c = x.contiguous()
        v_c = v.contiguous()

        # U: [H, D, R]  — pass directly (kernel expects this layout)
        if U.dim() == 2:
            # [D, R] → [1, D, R]
            U_k = U.unsqueeze(0).contiguous()
        else:
            U_k = U.contiguous()  # already [H, D, R]

        # W: [H, D, R]  — FIXED: do NOT mean-pool (that loses rank information)
        if W.dim() == 2:
            # [D, R] → [1, D, R]
            W_k = W.unsqueeze(0).contiguous()
        else:
            W_k = W.contiguous()  # already [H, D, R]

        # Physics kwargs with defaults
        clamp_val        = float(kwargs.get("clamp_val",        5.0))
        friction         = float(kwargs.get("friction",         0.0))
        vel_fric_scale   = float(kwargs.get("vel_fric_scale",   0.0))
        vel_sat          = float(kwargs.get("vel_sat",          0.0))
        sing_thresh      = float(kwargs.get("sing_thresh",      0.0))
        sing_strength    = float(kwargs.get("sing_strength",    0.0))
        enable_trace_norm = bool(kwargs.get("enable_trace_norm", True))
        is_paper_version  = bool(kwargs.get("is_paper_version",  False))

        # Optional gate tensors (friction gating)
        gate_w = kwargs.get("gate_w", torch.empty(0, device=x.device, dtype=x.dtype))
        gate_b = kwargs.get("gate_b", torch.empty(0, device=x.device, dtype=x.dtype))

        dt_tensor = torch.tensor(dt, dtype=x_c.dtype, device=x_c.device)

        # ── Single C++ call — kernel loops `steps` internally ─────────────
        result = f_leapfrog(
            x_c, v_c, U_k, W_k, force,
            dt_tensor, steps,
            clamp_val, friction, vel_fric_scale, vel_sat,
            gate_w, gate_b,
            sing_thresh, sing_strength,
            enable_trace_norm, is_paper_version,
        )

        if _DEBUG_SYNC:
            torch.cuda.synchronize()

        return result[0], result[1]

    except Exception:
        return None, None


def unified_yoshida_step(
    x: torch.Tensor,
    v: torch.Tensor,
    force: Optional[torch.Tensor],
    U: torch.Tensor,
    W: torch.Tensor,
    dt: float,
    steps: int = 1,
    **kwargs
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified Yoshida 4th-order integration step.
    Mirrors unified_leapfrog_step with the yoshida_fwd kernel.
    """
    if not is_cuda_active(v):
        return None, None

    _, f_yoshida = _get_cuda_integrators()
    if f_yoshida is None:
        return None, None

    try:
        x_c = x.contiguous()
        v_c = v.contiguous()
        U_k = U.unsqueeze(0).contiguous() if U.dim() == 2 else U.contiguous()
        W_k = W.unsqueeze(0).contiguous() if W.dim() == 2 else W.contiguous()

        clamp_val        = float(kwargs.get("clamp_val",        5.0))
        friction         = float(kwargs.get("friction",         0.0))
        vel_fric_scale   = float(kwargs.get("vel_fric_scale",   0.0))
        vel_sat          = float(kwargs.get("vel_sat",          0.0))
        sing_thresh      = float(kwargs.get("sing_thresh",      0.0))
        sing_strength    = float(kwargs.get("sing_strength",    0.0))
        enable_trace_norm = bool(kwargs.get("enable_trace_norm", True))
        is_paper_version  = bool(kwargs.get("is_paper_version",  False))

        gate_w = kwargs.get("gate_w", torch.empty(0, device=x.device, dtype=x.dtype))
        gate_b = kwargs.get("gate_b", torch.empty(0, device=x.device, dtype=x.dtype))

        dt_tensor = torch.tensor(dt, dtype=x_c.dtype, device=x_c.device)

        result = f_yoshida(
            x_c, v_c, U_k, W_k, force,
            dt_tensor, steps,
            clamp_val, friction, vel_fric_scale, vel_sat,
            gate_w, gate_b,
            sing_thresh, sing_strength,
            enable_trace_norm, is_paper_version,
        )

        if _DEBUG_SYNC:
            torch.cuda.synchronize()

        return result[0], result[1]

    except Exception:
        return None, None
