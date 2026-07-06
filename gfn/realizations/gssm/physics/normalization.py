"""
Centralized registry of geometry-dependent normalizations for manifold states.

Position normalization depends on topology, while velocity normalization is
applied in tangent space.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from ..constants import MAX_VELOCITY, EPSILON_STANDARD, TOPOLOGY_TORUS


class BaseManifoldNormalization(nn.Module, ABC):
    """Abstract base class for geometry-aware normalization layers."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TorusPositionNormalization(BaseManifoldNormalization):
    """
    Wraps position isometrically in [-π, π].
    Preserves toroidal topology: atan2(sin(x), cos(x)).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))


class TangentVelocityNormalization(nn.Module):
    """
    RMSNorm with maximum velocity clamp for tangent space.
    Prevents uncontrolled acceleration in high curvature regions.
    """
    def __init__(self, dim: int, eps: float = EPSILON_STANDARD):
        super().__init__()
        self.rms = nn.RMSNorm(dim, eps=eps)
        self.max_v = MAX_VELOCITY

    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Note: x is the velocity to normalize. context_x is position (optional for metric-aware)
        x = torch.clamp(x, -self.max_v, self.max_v)
        return self.rms(x)


class MetricAwareVelocityNormalization(nn.Module):
    """
    Normalization that scales velocity based on the Riemannian metric.
    Ensures the geodesic norm ||v||_g does not exceed the physical limit.
    """
    def __init__(self, dim: int, geometry=None, max_v: float = MAX_VELOCITY):
        super().__init__()
        self.geometry = geometry
        self.max_v = max_v
        self.rms = nn.RMSNorm(dim)  # Fallback if no geometry

    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.geometry is not None and context_x is not None:
            # x: [B, D] (velocity), context_x: [B, D] (position)
            # 1. Get metric tensor g(context_x)
            g = self.geometry.metric_tensor(context_x)  # [B, D, D] or [B, D] or [D, D]
            
            # 2. Compute squared norm: x^T g x
            # Use broadcast matmul for more robustness than bmm
            # v: [B, D] -> [B, 1, D]
            v_exp = x.unsqueeze(1)
            if g.dim() == 2 and g.shape[0] == x.shape[0]: # [B, D] - diagonal metric
                norm_sq = (x * g * x).sum(dim=-1, keepdim=True).unsqueeze(-1)
            elif g.dim() == 2: # [D, D] - constant metric
                g_exp = g.unsqueeze(0) # [1, D, D]
                norm_sq = v_exp @ g_exp @ v_exp.transpose(1, 2)
            else: # [B, D, D]
                norm_sq = v_exp @ g @ v_exp.transpose(1, 2)
            
            norm_g = torch.sqrt(norm_sq.squeeze(-1).squeeze(-1) + 1e-8)
            
            # 3. Scale if exceeds max_v
            scale = torch.clamp(self.max_v / norm_g, max=1.0)
            x = x * scale.unsqueeze(-1)
            
            # 4. Return clamped velocity with preserved magnitude
            return x
            
        # Fallback to standard clamp
        return torch.clamp(x, -self.max_v, self.max_v)


class EuclideanPositionNormalization(BaseManifoldNormalization):
    """
    Identity for Euclidean positions.
    Physically safer than RMSNorm for position coordinates.
    """
    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class IdentityNormalization(BaseManifoldNormalization):
    """Pass-through — use when no normalization is required."""
    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class ManifoldNormalizationRegistry:
    """
    Centralized registry to get appropriate normalization
    according to physical variable type and manifold topology.

    Available types:
      'position_torus'    — atan2(sin, cos) wrapping
      'position_euclidean'— Identity (safe for Euclidean coordinates)
      'velocity_tangent'  — Clamped RMSNorm (tangent space)
      'velocity_metric'   — MetricAware norm (strict)
      'feature_hidden'    — Same as velocity_tangent (internal features)
      'identity'          — No transformation
    """
    _REGISTRY = {
        'position_torus':    TorusPositionNormalization,
        'position_euclidean': EuclideanPositionNormalization,
        'velocity_tangent':  TangentVelocityNormalization,
        'velocity_metric':   MetricAwareVelocityNormalization,
        'feature_hidden':    TangentVelocityNormalization,
        'identity':          IdentityNormalization,
    }

    @classmethod
    def get(cls, norm_type: str, dim: int = 64, geometry=None) -> nn.Module:
        norm_cls = cls._REGISTRY.get(norm_type.lower(), IdentityNormalization)
        if norm_cls in (TangentVelocityNormalization,):
            return norm_cls(dim)
        if norm_cls == MetricAwareVelocityNormalization:
            return norm_cls(dim, geometry=geometry)
        return norm_cls()

    @classmethod
    def get_for_topology(cls, topology: str, dim: int = 64, 
                         is_velocity: bool = False, geometry=None) -> nn.Module:
        """
        Shortcut: automatically selects the correct normalization
        based on topology and whether it is position or velocity.
        """
        if is_velocity:
            # If geometry is available, we use the strict one
            if geometry is not None:
                return cls.get('velocity_metric', dim, geometry=geometry)
            return cls.get('velocity_tangent', dim)
        if topology.lower().strip() == TOPOLOGY_TORUS:
            return cls.get('position_torus', dim)
        return cls.get('position_euclidean', dim)


__all__ = [
    'ManifoldNormalizationRegistry',
    'TorusPositionNormalization',
    'TangentVelocityNormalization',
    'EuclideanPositionNormalization',
    'IdentityNormalization',
]
