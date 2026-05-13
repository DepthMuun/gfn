"""
gfn/math/differential.py — GFN V5
Álgebra diferencial y transporte.
"""
import torch

def christoffel_contraction(gamma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Contracción de los símbolos de Christoffel con el vector velocidad.
    γ(v) ≈ Γ^k_ij v^i v^j
    """
    return (gamma * v).sum(dim=-1, keepdim=True) * v

def parallel_transport_approx(v: torch.Tensor, gamma: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Transporte paralelo aproximado sin curvatura explícita.
    Δv ≈ -Γ(v,v)·dt
    """
    return v - christoffel_contraction(gamma, v) * dt
