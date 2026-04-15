"""
gfn/math/differential.py — GFN V5
Differential algebra and transport.
"""
import torch

def christoffel_contraction(gamma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Contraction of Christoffel symbols with velocity vector.
    γ(v) ≈ Γ^k_ij v^i v^j
    """
    return (gamma * v).sum(dim=-1, keepdim=True) * v

def parallel_transport_approx(v: torch.Tensor, gamma: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Approximate parallel transport without explicit curvature.
    Δv ≈ -Γ(v,v)·dt
    """
    return v - christoffel_contraction(gamma, v) * dt
