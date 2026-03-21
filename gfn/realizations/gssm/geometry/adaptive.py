"""
AdaptiveRiemannianGeometry — GFN V5
Adaptive rank Christoffel symbol decomposition.
Migrated from gfn/geo/riemannian/adaptive_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from ..constants import CURVATURE_CLAMP
from ..config.schema import PhysicsConfig
from ..geometry.base import BaseGeometry
from ..registry import register_geometry


@register_geometry('adaptive')
class AdaptiveRiemannianGeometry(BaseGeometry):
    """
    Adjusts the effective curvature rank dynamically based on velocity complexity.

    Architecture:
      eff_rank = f(||v||)  in [min_rank, max_rank]
      Γ(v) = W[:, :eff_rank] @ (U[:, :eff_rank]^T v)^2
    """

    def __init__(self, dim: int, max_rank: int = 64, config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self.dim = dim
        self.max_rank = max_rank
        self.min_rank_ratio = 0.1

        self.U_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
        self.W_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)

        # Complexity predictor: maps v → rank_ratio ∈ [0, 1]
        self.complexity_net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Initialize bias to start with a mostly-open rank to avoid vanishing gradients
        nn.init.constant_(self.complexity_net[-2].bias, 1.0)
        
        self.return_friction_separately = True

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        # 1. Predict rank weight (soft-mask)
        # Use complexity_net to predict a value p in [0, 1]
        p = self.complexity_net(v)  # [B, 1]
        
        # Create a soft mask for the rank dimension: [B, max_rank]
        # Mask[i] = sigmoid(slope * (p * max_rank - i))
        # This approximates hard-slicing but is differentiable.
        indices = torch.arange(self.max_rank, device=v.device).float()
        slope = 10.0
        soft_mask = torch.sigmoid(slope * (p * self.max_rank - indices)) # [B, max_rank]

        # 2. Christoffel using all components modulated by mask
        proj = torch.matmul(v, self.U_full)   # [B, max_rank]
        sq = proj * proj                     # [B, max_rank]
        modulated_sq = sq * soft_mask         # [B, max_rank]
        gamma = torch.matmul(modulated_sq, self.W_full.t())  # [B, dim]

        # 3. Friction (ensure mu is not just zero)
        # Fallback to config friction or a base value
        friction_base = getattr(self.config.stability, 'friction', 0.1)
        mu = torch.full_like(v, friction_base)

        gamma_clamped = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)

        if self.return_friction_separately:
            return gamma_clamped, mu

        return gamma_clamped + mu * v

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)
