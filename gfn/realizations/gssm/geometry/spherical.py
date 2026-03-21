import torch
from typing import Optional, Union, Tuple, Any
from ..geometry.base import BaseGeometry
from ..registry import register_geometry
from ..constants import EPS, TOPOLOGY_SPHERE

@register_geometry(TOPOLOGY_SPHERE)
class SphericalGeometry(BaseGeometry):
    """
    Spherical Geometry (Analytical).
    Computes Christoffel symbols for a constant positive curvature space.
    """
    def __init__(self, dim: int, rank: int = 16, config: Optional[Any] = None, **kwargs):
        super().__init__(config)
        self.dim = dim

    def christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        """Analytical Christoffel symbols for S^n are typically zero in standard embedding or complex in other charts."""
        # For our GFN purposes, we often use the simplified 'spherical_christoffel_torch' 
        # which acts as a centering/restoring force towards the sphere surface.
        return torch.zeros_like(x)

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, 
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)
            
        # Simplified analytical spherical coupling (restoring force)
        xv = torch.sum(x * v, dim=-1, keepdim=True)
        vv = torch.sum(v * v, dim=-1, keepdim=True)
        
        # Gamma = -(2.0 * xv * v - vv * x)
        gamma = -(2.0 * xv * v - vv * x)
        
        # Apply standard V5 clamping
        clamp_val = getattr(self, 'clamp_val', 5.0)
        return clamp_val * torch.tanh(gamma / clamp_val)

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Identity metric for simplified spherical chart."""
        return torch.ones_like(x)

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Great-circle distance approximation."""
        dot = torch.sum(x1 * x2, dim=-1)
        # Assuming points are on unit sphere
        return torch.acos(torch.clamp(dot, -1.0 + EPS, 1.0 - EPS))
