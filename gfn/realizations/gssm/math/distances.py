"""
gfn/math/geometry.py — GFN V5
Funciones base de geometría Riemanniana y distancias.
"""
import torch

def geodesic_distance_torus(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Distancia geodésica en el toro — wrapping angular."""
    diff = x1 - x2
    return torch.norm(torch.atan2(torch.sin(diff), torch.cos(diff)), dim=-1)

def geodesic_distance_euclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Distancia L2 estándar."""
    return torch.norm(x1 - x2, dim=-1)

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap ángulos a [-π, π]."""
    return torch.atan2(torch.sin(x), torch.cos(x))
