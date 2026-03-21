"""
HolographicRiemannianGeometry — GFN V5
AdS/CFT-inspired holographic extensions (Paper 18).
Migrated from gfn/geo/physical/holographic_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from ..config.schema import PhysicsConfig
from ..geometry.base import BaseGeometry
from ..registry import register_geometry


@register_geometry('holographic')
class HolographicRiemannianGeometry(BaseGeometry):
    """
    Conformal manifold inspired by Bulk-Boundary (AdS/CFT) correspondence.

    Lifts boundary state x → bulk (x, z) where z is the holographic radial dim.
    Conformal metric: g_ij = (1/z(x)²) · δ_ij

    The Christoffel correction adds an AdS-geodesic term to any base geometry.
    """

    def __init__(self, base_geometry: BaseGeometry, z_min: float = 0.1,
                 z_max: float = 10.0, config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self.base_geometry = base_geometry
        self.dim = getattr(base_geometry, 'dim', None)
        self.z_min = z_min
        self.z_max = z_max

        dim = self.dim or 0
        if dim > 0:
            self.radial_net: nn.Module = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, 1),
                nn.Softplus()
            )
        else:
            self.radial_net = nn.Identity()

    def get_z_and_grad(self, x: torch.Tensor):
        x_req = x.detach().requires_grad_(True)
        with torch.enable_grad():
            z = self.radial_net(x_req) + self.z_min
            z = torch.clamp(z, max=self.z_max)
            grad_z = torch.autograd.grad(
                z.sum(), x_req,
                create_graph=self.training,
                retain_graph=False
            )[0]
        return z, grad_z

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.get_z_and_grad(x)
        return (1.0 / z.pow(2)) * torch.ones_like(x)

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out_base = self.base_geometry(x, v, force=force, **kwargs)
        if isinstance(out_base, tuple):
            gamma_base, mu = out_base
        else:
            gamma_base, mu = out_base, torch.zeros_like(v) if v is not None else torch.zeros_like(x)

        if v is None:
            if self.return_friction_separately:
                return gamma_base, mu
            return gamma_base

        z, grad_z = self.get_z_and_grad(x)
        v_dot_gradz = (v * grad_z).sum(dim=-1, keepdim=True)
        v_sq = (v * v).sum(dim=-1, keepdim=True)
        gamma_ads = -(1.0 / z) * (2.0 * v_dot_gradz * v - v_sq * grad_z)

        gamma_total = gamma_base + gamma_ads
        
        if self.return_friction_separately:
            return gamma_total, mu
            
        return gamma_total + mu * v

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_geometry.project(x)

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.base_geometry.dist(x1, x2)
