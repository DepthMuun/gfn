"""
gfn/math/physics.py — GFN V5
Curvature and mechanical energy metrics.
"""
import torch

def ricci_scalar_approx(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Approximation of Ricci scalar from low-rank decomposition.
    R ≈ tr(W^T W) / dim
    """
    return (W * W).sum() / (W.shape[0] + 1e-8)

def hamiltonian_energy(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Kinetic energy H = 0.5 * ||v||²."""
    return 0.5 * (v ** 2).sum(dim=-1)
