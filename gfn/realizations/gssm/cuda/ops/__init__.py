"""
gfn/cuda/ops/__init__.py
Exports all fused CUDA operations.
Gracefully returns None for any op whose kernel is not compiled.
"""
from ...cuda import CUDA_AVAILABLE
import os
import sys

# Ensure gfn/csrc is in PYTHONPATH to find the compiled .pyd
_csrc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "csrc"))
if _csrc_path not in sys.path:
    sys.path.insert(0, _csrc_path)

def _get_op(module_path: str, name: str):
    """Safely import a CUDA binding, returning None on failure."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name, None)
    except Exception:
        return None

# ── Geometry ──────────────────────────────────────────────────────────────────
christoffel_cuda_fwd = _get_op("gfn_cuda", "compute_christoffel_symbols_fwd")
christoffel_cuda_bwd = _get_op("gfn_cuda", "compute_christoffel_symbols_bwd")
low_rank_christoffel_fwd = _get_op("gfn_cuda", "low_rank_christoffel_fwd")
low_rank_christoffel_bwd = _get_op("gfn_cuda", "low_rank_christoffel_bwd")
toroidal_christ_fwd = _get_op("gfn_cuda", "toroidal_geo_christoffel_fwd")

# ── Integrators ───────────────────────────────────────────────────────────────
heun_fused       = _get_op("gfn_cuda", "heun_fwd")
leapfrog_fused   = _get_op("gfn_cuda", "leapfrog_fwd")
yoshida_fused    = _get_op("gfn_cuda", "yoshida_fwd")
rk4_fused        = _get_op("gfn_cuda", "rk4_fwd")

# ── Loss ──────────────────────────────────────────────────────────────────────
toroidal_loss_fwd = _get_op("gfn_cuda", "toroidal_distance_loss_fwd")
toroidal_loss_bwd = _get_op("gfn_cuda", "toroidal_distance_loss_bwd")

def __getattr__(name):
    if name.endswith(("_fused", "_fwd", "_bwd", "_cuda")):
        return None
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "CUDA_AVAILABLE",
    "christoffel_cuda_fwd", "christoffel_cuda_bwd",
    "low_rank_christoffel_fwd", "low_rank_christoffel_bwd",
    "toroidal_christ_fwd", "heun_fused", "leapfrog_fused",
    "yoshida_fused", "rk4_fused", "toroidal_loss_fwd", "toroidal_loss_bwd"
]
