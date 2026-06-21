"""
Intrinsic exploration force that repels states from dense regions of the batch.
"""

import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN

class GeometricCuriosityForce(nn.Module):
    """
    Applies a repulsive force from dense geometric aggregates,
    using local curvature or position history.
    """
    def __init__(self, strength: float = 0.1, decay: float = 0.99):
        super().__init__()
        self.strength = strength
        self.decay = decay
        # For simplified exploration, we repel from the batch center of mass
        # More advanced options would require an external Density Estimator.
        
    def forward(self, x: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the repulsive curiosity force based on the current batch spread.
        """
        if self.strength <= 0.0:
            return torch.zeros_like(v)
        
        # 1. Find the trivial batch 'attractor' (Center of gravity)
        if kwargs.get('topology', TOPOLOGY_EUCLIDEAN) == TOPOLOGY_TORUS:
            # Mean Circular: atan2(mean(sin), mean(cos))
            sin_x = torch.sin(x); cos_x = torch.cos(x)
            batch_center = torch.atan2(sin_x.mean(dim=0, keepdim=True), cos_x.mean(dim=0, keepdim=True))
            # 2. Escape vector (geodesic direction)
            direction = x - batch_center
            direction = torch.atan2(torch.sin(direction), torch.cos(direction))
        else:
            batch_center = x.mean(dim=0, keepdim=True)
            # 2. Escape vector (Euclidean direction)
            direction = x - batch_center
        
        # 3. Force inversely proportional to distance
        dist_sq = (direction ** 2).sum(dim=-1, keepdim=True) + 1e-6
        repulsion_mag = self.strength / dist_sq
        
        # 4. Normalize direction and scale
        force = (direction / (dist_sq ** 0.5 + 1e-8)) * repulsion_mag
        
        # Limit maximum force to prevent instability
        return torch.clamp(force, min=-5.0, max=5.0)
