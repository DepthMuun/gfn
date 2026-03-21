"""
HyperRiemannianGeometry — GFN V5
Context-dependent (gated) Christoffel symbols.
Migrated from gfn/geo/topological/hyperbolic_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from ..constants import CURVATURE_CLAMP, EPS, TOPOLOGY_TORUS
from ..config.schema import PhysicsConfig
from ..geometry.low_rank import LowRankRiemannianGeometry
from ..registry import register_geometry


@register_geometry('hyperbolic')
class HyperRiemannianGeometry(LowRankRiemannianGeometry):
    """
    Hyper-Christoffel: geometry conditioned on current position.

    Architecture:
      U(x) = U_static * diag(Gate_u(x))   — position-scaled basis
      W(x) = W_static * diag(Gate_w(x))
      Γ(v | x) = W(x) @ (U(x)^T v)²

    Gates output values in [0, 2] initialized near 1.0 (identity).
    """

    def __init__(self, dim: int, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(dim, rank, num_heads=num_heads, config=config)
        self.return_friction_separately = True

        self.gate_u = nn.Linear(dim, rank)
        self.gate_w = nn.Linear(dim, rank)
        nn.init.zeros_(self.gate_u.weight); nn.init.zeros_(self.gate_u.bias)
        nn.init.zeros_(self.gate_w.weight); nn.init.zeros_(self.gate_w.bias)

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        original_shape = v.shape
        # Handle multi-head [B, H, HD] -> [B*H, HD]
        if v.dim() == 3:
            B, H, HD = v.shape
            v_flat = v.reshape(B * H, HD)
            x_flat = x.reshape(B * H, HD)
        else:
            v_flat = v
            x_flat = x
            B, H = None, None

        # Context gates in [0, 2]
        g_u = torch.sigmoid(self.gate_u(x_flat)) * 2.0  # [B*H, rank]
        g_w = torch.sigmoid(self.gate_w(x_flat)) * 2.0

        # Modulate static basis
        # self.U is [HD, rank] or [H, HD, rank]
        U_eff = self.U if self.U.dim() == 2 else self.U.mean(0)
        proj_static = torch.matmul(v_flat, U_eff) # [B*H, rank]
        proj_dynamic = proj_static * g_u

        # Soft-saturation to prevent energy explosion
        sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic) + EPS)
        sq_modulated = sq_dynamic * g_w

        W_t = self.W.t() if self.W.dim() == 2 else self.W.mean(0).t()
        gamma = torch.matmul(sq_modulated, W_t) # [B*H, HD]

        # Restore original shape if multi-head
        if B is not None:
            gamma = gamma.view(original_shape)
            x_flat_for_mu = x_flat
            v_flat_for_mu = v_flat
        else:
            x_flat_for_mu = x
            v_flat_for_mu = v

        # Friction (position-dependent, gated)
        x_in = torch.cat([torch.sin(x_flat), torch.cos(x_flat)], dim=-1) \
            if self.topology_type == TOPOLOGY_TORUS else x_flat
        mu = self.friction_gate(x_in, force=force)
        
        if mu.shape != v.shape:
            mu = mu.view_as(v) if mu.numel() == v.numel() else mu.mean(dim=-1, keepdim=True)

        gamma = self._normalize(gamma)
        gamma = self.clamp_val * torch.tanh(gamma / self.clamp_val)

        if self.return_friction_separately:
            return gamma, mu

        return gamma + mu * v
