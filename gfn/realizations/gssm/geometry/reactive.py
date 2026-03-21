"""
ReactiveRiemannianGeometry — GFN V5
Active-inference geometry: curvature reacts to system state.
Migrated from gfn/geo/physical/reactive_field_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any

from ..constants import CURVATURE_CLAMP, EPS, DEFAULT_PLASTICITY, TOPOLOGY_TORUS
from ..config.schema import PhysicsConfig
from ..geometry.low_rank import LowRankRiemannianGeometry
from ..registry import register_geometry

# Default constants
SINGULARITY_THRESHOLD = 0.5
BLACK_HOLE_STRENGTH = 3.0
SINGULARITY_GATE_SLOPE = 10.0


@register_geometry('reactive')
class ReactiveRiemannianGeometry(LowRankRiemannianGeometry):
    """
    Geometry that reacts to the system's own state via active inference.

    Enhancements over LowRank:
    1. Plasticity: Christoffel symbols scaled by kinetic energy (curv. amplification ≈ attention).
    2. Singularities: Soft curvature amplification near semantic attractors.

    Note: These are regularization/attention mechanisms, NOT physical manifold properties.
    """

    def __init__(self, dim: int, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(dim, rank, num_heads=num_heads, config=config)
        self.return_friction_separately = True

        self.active_cfg = self.config.active_inference
        self.plasticity = getattr(self.active_cfg, 'plasticity', DEFAULT_PLASTICITY)

        sing_cfg = self.config.singularities
        sing_enabled = getattr(sing_cfg, 'enabled', False)

        if sing_enabled:
            self.semantic_certainty_threshold = getattr(sing_cfg, 'threshold', SINGULARITY_THRESHOLD)
            self.curvature_amplification_factor = getattr(sing_cfg, 'strength', BLACK_HOLE_STRENGTH)
            gate_input_dim = dim * 2 if self.topology == TOPOLOGY_TORUS else dim
            if num_heads > 1:
                self.V_weight = nn.Parameter(torch.zeros(num_heads, gate_input_dim, 1))
            else:
                self.V = nn.Linear(gate_input_dim, 1)
                nn.init.zeros_(self.V.weight)
                nn.init.constant_(self.V.bias, -2.0)  # Start gate closed
        else:
            self.semantic_certainty_threshold = SINGULARITY_THRESHOLD
            self.curvature_amplification_factor = BLACK_HOLE_STRENGTH
            self.V = None

    def _get_potential(self, x_in: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute singularity potential, returns None if disabled."""
        if not getattr(self.config.singularities, 'enabled', False):
            return None
        if self.num_heads > 1:
            return torch.sigmoid(torch.matmul(x_in.unsqueeze(-2), self.V_weight).squeeze(-2))
        elif self.V is not None:
            return torch.sigmoid(self.V(x_in))
        return None

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        # 1. Base curvature from LowRank
        res = super().forward(x, v, force=force, **kwargs)
        if isinstance(res, tuple):
            gamma, mu = res
        else:
            gamma, mu = res, torch.zeros_like(v) if v is not None else torch.zeros_like(x)

        if not self.active_cfg.enabled:
            if self.return_friction_separately:
                return gamma, mu
            return gamma + mu * v if v is not None else gamma

        # 2. Plasticity: scale curvature by kinetic energy
        react_cfg = self.active_cfg.reactive_curvature
        react_enabled = react_cfg.get('enabled', False) if isinstance(react_cfg, dict) else False
        if react_enabled and self.plasticity > 0.0:
            energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
            gamma = gamma * (1.0 + self.plasticity * energy)

        # 3. Singularity amplification
        if getattr(self.config.singularities, 'enabled', False) and x is not None:
            x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if self.topology == TOPOLOGY_TORUS else x
            potential = self._get_potential(x_in)
            if potential is not None:
                gate_slope = getattr(self.config.singularities, 'gate_slope', SINGULARITY_GATE_SLOPE)
                is_amplified = torch.sigmoid(gate_slope * (potential - self.semantic_certainty_threshold))
                amp = 1.0 + is_amplified * (self.curvature_amplification_factor - 1.0)
                gamma = gamma * amp
                limit = self.curvature_amplification_factor * CURVATURE_CLAMP
                gamma = limit * torch.tanh(gamma / limit)

        if self.return_friction_separately:
            return gamma, mu

        return gamma + mu * v if v is not None else gamma
