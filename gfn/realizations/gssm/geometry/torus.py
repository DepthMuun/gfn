"""
ToroidalRiemannianGeometry — GFN V5
Full analytical torus geometry (canonical implementation).
Replaces old stub. Migrated from gfn/geo/topological/toroidal_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any

from ..constants import (
    EPS, TOPOLOGY_TORUS, CURVATURE_CLAMP,
    DEFAULT_FRICTION, GATE_BIAS_CLOSED
)
from ..config.schema import PhysicsConfig
from ..geometry.base import BaseGeometry
from ..registry import register_geometry

# Import modular friction component
from ..physics.components.friction import FrictionGate

# Torus-specific constants
TOROIDAL_CURVATURE_SCALE = 0.1
CLAMP_MIN_STRONG = 1e-4
EPSILON_SMOOTH = 1e-9

try:
    import toroidal_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False



@register_geometry(TOPOLOGY_TORUS)
class ToroidalRiemannianGeometry(BaseGeometry):
    """
    Curved torus geometry with exact (analytical) Christoffel symbols.

    Metric (2D torus paired dimensions):
      g_theta = r²
      g_phi   = (R + r·cos θ)²

    Generalizes to N dims by pairing (theta_0,phi_0), (theta_1,phi_1), ...
    """

    def __init__(self, dim: int = 64, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self.dim = dim
        self.num_heads = num_heads

        topo = self.config.topology
        
        # CORRECCIÓN: Hacer R y r aprendibles según GFN_Paper_Complete.md Sección 5.1
        # "where R_i are the (learnable) radii"
        # Y 01_HYPER_TORUS.md Sección 2.2: R es radio mayor, r es radio menor
        learnable_R = getattr(topo, 'learnable_R', True)
        learnable_r = getattr(topo, 'learnable_r', True)
        
        if learnable_R:
            # R como parámetro aprendible
            self.R = nn.Parameter(torch.tensor(topo.R, dtype=torch.float32))
        else:
            # R como buffer no-entrenable
            self.register_buffer('R', torch.tensor(topo.R, dtype=torch.float32))
            
        if learnable_r:
            # r como parámetro aprendible
            self.r = nn.Parameter(torch.tensor(topo.r, dtype=torch.float32))
        else:
            # r como buffer no-entrenable
            self.register_buffer('r', torch.tensor(topo.r, dtype=torch.float32))
        
        self.topology = topo.type.lower()
        self.clamp_val = self.config.stability.curvature_clamp

        active_cfg = self.config.active_inference
        self.active_cfg = active_cfg
        rc = active_cfg.reactive_curvature
        self.plasticity = rc.get('plasticity', getattr(active_cfg, 'plasticity', 0.0)) \
            if isinstance(rc, dict) else getattr(active_cfg, 'plasticity', 0.0)

        sing_cfg = self.config.singularities
        self.singularity_threshold = sing_cfg.threshold
        self.black_hole_strength = sing_cfg.strength

        gate_input_dim = dim * 2  # [sin(x), cos(x)]
        friction_mode = getattr(self.config.stability, 'friction_mode', 'static')
        self.friction_gate = FrictionGate(
            dim, gate_input_dim=gate_input_dim, mode=friction_mode, num_heads=num_heads
        )

        # Optional singularity potential gate
        if getattr(sing_cfg, 'enabled', False):
            if num_heads > 1:
                self.V_weight = nn.Parameter(torch.zeros(num_heads, gate_input_dim, 1))
                self.V_bias = nn.Parameter(torch.full((num_heads, 1), GATE_BIAS_CLOSED))
            else:
                self.V = nn.Linear(gate_input_dim, 1)
                nn.init.zeros_(self.V.weight)
                nn.init.constant_(self.V.bias, GATE_BIAS_CLOSED)
        else:
            self.V = None

    def validate_dimensions(self, x: torch.Tensor):
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"ToroidalGeometry requires even dim for (θ,φ) pairing. Got {x.shape[-1]}"
            )

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_dimensions(x)
        g = torch.ones_like(x)
        for i in range(0, x.shape[-1], 2):
            th = x[..., i]
            g[..., i] = self.r ** 2
            g[..., i + 1] = (self.R + self.r * torch.cos(th)) ** 2
        return g

    def connection(self, v: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        gamma = torch.zeros_like(v)
        for i in range(0, (d // 2) * 2, 2):
            th = x[..., i]
            v_th, v_ph = v[..., i], v[..., i + 1]
            w_th, w_ph = w[..., i], w[..., i + 1]
            denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)
            term_th = (denom * torch.sin(th) / (self.r + EPSILON_SMOOTH)) * TOROIDAL_CURVATURE_SCALE
            gamma[..., i] = term_th * (v_ph * w_ph)
            term_ph = (-(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)) * TOROIDAL_CURVATURE_SCALE
            gamma[..., i + 1] = term_ph * (v_ph * w_th + v_th * w_ph)
        return gamma

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        self.validate_dimensions(x)

        if HAS_CUDA_EXT and x.is_cuda and v is not None and v.is_cuda:
            gamma = toroidal_cuda.forward(
                x, v, self.R, self.r, 
                TOROIDAL_CURVATURE_SCALE, EPSILON_SMOOTH, CLAMP_MIN_STRONG
            )
        else:
            is_odd = (self.dim % 2 != 0)
            x_pad = torch.nn.functional.pad(x, (0, 1)) if is_odd else x
            v_pad = torch.nn.functional.pad(v, (0, 1)) if is_odd else v
    
            th = x_pad[..., 0::2]
            v_th = v_pad[..., 0::2]
            v_ph = v_pad[..., 1::2]
    
            denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)
            term_th = (denom * torch.sin(th) / (self.r + EPSILON_SMOOTH))
            term_ph = -(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)
    
            gamma_th = term_th * (v_ph ** 2) * TOROIDAL_CURVATURE_SCALE
            gamma_ph = 2.0 * term_ph * v_ph * v_th * TOROIDAL_CURVATURE_SCALE
    
            half = x.shape[-1] // 2
            gamma = torch.zeros_like(x)
            gamma[..., 0::2] = gamma_th[..., :half + x.shape[-1] % 2]
            gamma[..., 1::2] = gamma_ph[..., :half]

        # Friction gate
        x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        mu = self.friction_gate(x_in, force=force)
        if mu.shape[-1] != v.shape[-1]:
            if mu.shape[-1] == 2 * v.shape[-1]:
                mu = (mu[..., :v.shape[-1]] + mu[..., v.shape[-1]:]) / 2.0

        # Active inference
        if self.active_cfg.enabled:
            rc = self.active_cfg.reactive_curvature
            react_enabled = rc.get('enabled', False) if isinstance(rc, dict) else False
            if react_enabled and self.plasticity > 0.0:
                energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
                gamma = gamma * (1.0 + self.plasticity * energy)

            if getattr(self.config.singularities, 'enabled', False):
                if self.num_heads > 1:
                    potential = torch.sigmoid(
                        torch.matmul(x_in.unsqueeze(-2), self.V_weight).squeeze(-2)
                        + self.V_bias
                    )
                elif self.V is not None:
                    potential = torch.sigmoid(self.V(x_in))
                else:
                    potential = None

                if potential is not None:
                    soft_m = torch.sigmoid(5.0 * (potential - self.singularity_threshold))
                    gamma = gamma * (1.0 + soft_m * (self.black_hole_strength - 1.0))

        # CONTRACT: Always return (gamma_pure, mu) — engine applies friction, not geometry.
        # This unifies the contract with LowRankRiemannianGeometry (P0.2 fix).
        return gamma, mu

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = x1 - x2
        return torch.norm(torch.atan2(torch.sin(diff), torch.cos(diff)), dim=-1)


@register_geometry('flat_torus')
class FlatToroidalRiemannianGeometry(BaseGeometry):
    def __init__(self, dim: int = 64, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self.dim = dim
        self.num_heads = num_heads
        topo = self.config.topology
        
        # CORRECCIÓN: Hacer R y r aprendibles también en FlatTorus
        learnable_R = getattr(topo, 'learnable_R', True)
        learnable_r = getattr(topo, 'learnable_r', True)
        
        if learnable_R:
            self.R = nn.Parameter(torch.tensor(topo.R, dtype=torch.float32))
        else:
            self.register_buffer('R', torch.tensor(topo.R, dtype=torch.float32))
            
        if learnable_r:
            self.r = nn.Parameter(torch.tensor(topo.r, dtype=torch.float32))
        else:
            self.register_buffer('r', torch.tensor(topo.r, dtype=torch.float32))
            
        self.topology = topo.type.lower()
        self.return_friction_separately = True
        gate_input_dim = dim * 2
        friction_mode = getattr(self.config.stability, 'friction_mode', 'static')
        self.friction_gate = FrictionGate(
            dim, gate_input_dim=gate_input_dim, mode=friction_mode, num_heads=num_heads
        )

    def validate_dimensions(self, x: torch.Tensor):
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"FlatToroidalGeometry requires even dim for (θ,φ) pairing. Got {x.shape[-1]}"
            )

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_dimensions(x)
        return torch.ones_like(x)

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        self.validate_dimensions(x)
        x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        mu = self.friction_gate(x_in, force=force)
        if mu.shape[-1] != v.shape[-1]:
            if mu.shape[-1] == 2 * v.shape[-1]:
                mu = (mu[..., :v.shape[-1]] + mu[..., v.shape[-1]:]) / 2.0
            else:
                mu = mu.expand_as(v)
        gamma = torch.zeros_like(v)
        if self.return_friction_separately:
            return gamma, mu
        return gamma + mu * v

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = x1 - x2
        return torch.norm(torch.atan2(torch.sin(diff), torch.cos(diff)), dim=-1)
