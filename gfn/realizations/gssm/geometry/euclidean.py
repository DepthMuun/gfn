import torch
from .base import BaseGeometry
from ..registry import register_geometry
from ..constants import TOPOLOGY_EUCLIDEAN

@register_geometry(TOPOLOGY_EUCLIDEAN)
class EuclideanGeometry(BaseGeometry):
    """Standard Euclidean Space (Flat)."""
    
    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)
        
    def christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.norm(x1 - x2, dim=-1)
