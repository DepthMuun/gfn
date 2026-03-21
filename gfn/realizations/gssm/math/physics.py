"""
gfn/math/physics.py — GFN V5
Métricas de curvatura y energía mecánica.
"""
import torch

def ricci_scalar_approx(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Aproximación del escalar de Ricci desde la descomposición de rango bajo.
    R ≈ tr(W^T W) / dim
    """
    return (W * W).sum() / (W.shape[0] + 1e-8)

def hamiltonian_energy(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Energía cinética H = 0.5 * ||v||²."""
    return 0.5 * (v ** 2).sum(dim=-1)
