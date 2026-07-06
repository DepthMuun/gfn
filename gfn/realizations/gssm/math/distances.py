"""
Distance and wrapping helpers for Euclidean and toroidal coordinates.
"""
import torch

def geodesic_distance_torus(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on torus — angular wrapping."""
    diff = x1 - x2
    return torch.norm(torch.atan2(torch.sin(diff), torch.cos(diff)), dim=-1)

def geodesic_distance_euclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Standard L2 distance."""
    return torch.norm(x1 - x2, dim=-1)

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-π, π]."""
    return torch.atan2(torch.sin(x), torch.cos(x))
