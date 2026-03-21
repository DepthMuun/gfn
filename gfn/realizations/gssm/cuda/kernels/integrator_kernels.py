"""
Integrator Kernels — GFN V5
Unified entry points for numerical integration with hardware dispatching.
"""

import torch
from typing import Optional, Tuple, Any, Callable
from ...cuda import is_cuda_active

# Lazy imports for CUDA kernels
_euler_fused = None
_rk4_fused = None
_leapfrog_fused = None

def _get_cuda_integrators():
    global _euler_fused, _rk4_fused, _leapfrog_fused
    if _euler_fused is None:
        try:
            from ...cuda.ops import euler_fused, rk4_fused, leapfrog_fused
            _euler_fused = euler_fused
            _rk4_fused = rk4_fused
            _leapfrog_fused = leapfrog_fused
        except ImportError:
            pass
    return _euler_fused, _rk4_fused, _leapfrog_fused

def unified_leapfrog_step(
    x: torch.Tensor, 
    v: torch.Tensor, 
    force: Optional[torch.Tensor],
    U: torch.Tensor,
    W: torch.Tensor,
    dt: float,
    steps: int = 1,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified Leapfrog integration step."""
    if is_cuda_active(v):
        _, _, f_leapfrog = _get_cuda_integrators()
        if f_leapfrog is not None:
            try:
                # Prep parameters for CUDA kernel
                topo_id = kwargs.get('topology_id', 0)
                R = kwargs.get('R', 2.0)
                r = kwargs.get('r', 1.0)
                H = x.shape[1] if x.dim() == 3 else 1
                
                # U, W transformations
                if U.dim() == 2:
                    # [D, R] -> [1, R, D] -> [H, R, D]
                    U_k = U.T.unsqueeze(0).expand(H, -1, -1).contiguous()
                else:
                    # [H, D, R] -> [H, R, D]
                    U_k = U.transpose(1, 2).contiguous()
                    
                if W.dim() == 2:
                    # [D, R] -> [1, R] -> [H, R]
                    # Use mean instead of sum to preserve effective force scale
                    W_k = W.mean(dim=0).unsqueeze(0).expand(H, -1).contiguous()
                else:
                    # [H, D, R] -> [H, R]
                    W_k = W.abs().mean(dim=1).contiguous()
                
                cx, cv = x, v
                for _ in range(steps):
                    cx, cv = f_leapfrog(U_k, W_k, cx, cv, force, float(dt), int(topo_id), float(R), float(r), 0.0)
                return cx, cv
            except Exception:
                pass

    # Python fallback is handled by the higher-level Integrator classes in gfn/integrators/
    # This unified layer is primarily for hardware acceleration.
    return None, None # Signal fallback
