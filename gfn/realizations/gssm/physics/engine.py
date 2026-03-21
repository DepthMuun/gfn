"""
ManifoldPhysicsEngine — GFN V5
Central orchestration of physical forces.
Migrated and simplified from gfn/nn/physics/engine.py
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, Union

from ..interfaces.geometry import Geometry
from ..interfaces.physics import PhysicsEngine
from ..config.schema import PhysicsConfig
from ..constants import TOPOLOGY_TORUS, EPS, TOPOLOGY_EUCLIDEAN

from .components import (
    SingularityGate,
    HysteresisModule,
    BrownianForce,
    OUDynamicsForce,
    GeometricCuriosityForce,
)


class ManifoldPhysicsEngine(nn.Module):
    """
    Manifold Physics Engine — GFN V5.

    Computes the net acceleration:
      dv/dt = -Γ(x,v) + F_ext + F_friction + F_ghost

    Where:
      Γ(x,v)    = Christoffel-induced force (from geometry)
      F_ext     = External input force (token embedding signal)
      F_friction = -μ·v from geometry's friction gate or fallback config
      F_ghost   = Ghost force from hysteresis module (if enabled)

    The engine dispatches to `geometry.forward()` and interprets
    the return type (tensor or (tensor, friction) tuple).
    
    Supports optional singularity handling and hysteresis dynamics
    through modular components.
    """

    def __init__(
        self, 
        geometry: nn.Module,
        config: Optional[PhysicsConfig] = None,
        dim: Optional[int] = None,
        heads: int = 1
    ):
        super().__init__()
        self.geometry = geometry
        self.config = config or PhysicsConfig()

        topo = self.config.topology.type.lower()
        self.is_torus = (topo == TOPOLOGY_TORUS)
        # P0.2: engine is the SINGLE authority on friction application.
        # friction_fallback is used ONLY when geometry returns a plain tensor (no mu).
        self.friction_fallback = self.config.stability.friction
        self.velocity_friction_scale = getattr(self.config.stability, 'velocity_friction_scale', 0.0)
        # P2.3: velocity saturation via differentiable tanh clamp (0 = disabled)
        self.velocity_saturation = getattr(self.config.stability, 'velocity_saturation', 0.0)
        
        # Topology ID for hysteresis feature extraction
        self.topo_id = 1 if self.is_torus else 0
        
        # Initialize singularity gate if enabled
        self.singularity_gate: Optional[SingularityGate] = None
        if self.config.singularities.enabled:
            self.singularity_gate = SingularityGate.from_config(self.config.singularities)
            
        # Initialize hysteresis module if enabled
        self.hysteresis: Optional[HysteresisModule] = None
        if self.config.hysteresis.enabled and dim is not None:
            self.hysteresis = HysteresisModule.from_config(
                self.config.hysteresis, 
                dim=dim, 
                heads=heads
            )
            
        # Initialize Stochastic Forces
        self.stochasticity_module: Optional[nn.Module] = None
        stoch_cfg = getattr(self.config.active_inference, 'stochasticity', {})
        if stoch_cfg.get('enabled', False):
            if stoch_cfg.get('type') == 'brownian':
                self.stochasticity_module = BrownianForce(sigma=stoch_cfg.get('sigma', 0.01))
            elif stoch_cfg.get('type') == 'ou':
                self.stochasticity_module = OUDynamicsForce(
                    sigma=stoch_cfg.get('sigma', 0.01),
                    theta=stoch_cfg.get('theta', 0.15),
                    mu=stoch_cfg.get('mu', 0.0)
                )

        # Initialize Curiosity Exploration
        self.curiosity_module: Optional[GeometricCuriosityForce] = None
        curiosity_cfg = getattr(self.config.active_inference, 'curiosity', {})
        if curiosity_cfg.get('enabled', False):
            self.curiosity_module = GeometricCuriosityForce(
                strength=curiosity_cfg.get('strength', 0.1),
                decay=curiosity_cfg.get('decay', 0.99)
            )

    def compute_acceleration(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        force: Optional[torch.Tensor] = None,
        metric_component: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute net acceleration from all physical contributions.

        Args:
            x:     Position [B, H, D] or [B, D]
            v:     Velocity, same shape as x
            force: External force (token/sequence signal), same shape
            metric_component: Optional metric component for singularity damping
        Returns:
            Acceleration tensor (same shape as v)
        """
        if x.shape != v.shape:
            raise ValueError(f"[PhysicsEngine] Shape mismatch: x={x.shape} vs v={v.shape}")

        # 1. Geometry: pure Christoffel symbols Γ(x,v)
        #    CONTRACT: geometry returns EITHER:
        #      - (gamma, mu): Christoffel + friction coefficient (preferred)
        #      - gamma: Christoffel only (legacy geometries, use friction_fallback)
        geo_out = self.geometry(x, v, force=force, **kwargs)

        if isinstance(geo_out, tuple):
            # Geometry provided Γ and potentially an excess friction coefficient μ_geo (gated)
            christoffel, mu_geo = geo_out
        else:
            christoffel = geo_out
            mu_geo = 0.0

        # 1.5. Centralized Friction Logic
        mu_total = self.get_friction_coefficient(x, v, mu_geo=mu_geo)
        friction_term = mu_total * v

        # 2. Net acceleration: -Γ - friction_term
        net_accel = -christoffel - friction_term

        # 3. Accumulate External and Secondary Forces
        if force is not None:
            net_accel = net_accel + force

        # 4. Apply hysteresis ghost force if enabled
        if self.hysteresis is not None:
            ghost_force = self.hysteresis(x, v, topo_id=self.topo_id)
            net_accel = net_accel + ghost_force
            
        # 5. Apply Stochastic Forces / Langevin Noise (if enabled)
        if self.stochasticity_module is not None and dt is not None:
            stoch_force = self.stochasticity_module(x, v, dt)
            net_accel = net_accel + stoch_force
            
        # 6. Apply Geometric Curiosity (Exploration) (if enabled)
        if self.curiosity_module is not None:
            curiosity_force = self.curiosity_module(x, v, **kwargs)
            net_accel = net_accel + curiosity_force

        # ── P0.4 Architecture Fix: Singularity Damping ───────────────────────
        # Apply damping AFTER all forces have been accumulated. This creates a 
        # "safety bubble" that prevents any force from launching the state into
        # divergent regions (where Christoffel symbols blow up).
        if self.singularity_gate is not None and metric_component is not None:
            net_accel = self.singularity_gate.damp_force(net_accel, metric_component)

        return net_accel

    def validate_state(self, x: torch.Tensor, v: torch.Tensor) -> None:
        """Validates that state tensors are within physical boundaries."""
        if torch.isnan(x).any() or torch.isnan(v).any():
            from ..errors import PhysicsError
            raise PhysicsError("NaN detected in physics state.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_acceleration(*args, **kwargs)
    
    def apply_singularity_damping(self, v: torch.Tensor, metric_component: torch.Tensor) -> torch.Tensor:
        """
        Apply singularity-based velocity damping.
        
        Args:
            v: Velocity tensor
            metric_component: Metric component for singularity detection
        Returns:
            Damped velocity
        """
        if self.singularity_gate is not None:
            return self.singularity_gate.damp_velocity(v, metric_component)
        return v
    
    def get_ghost_force(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Get ghost force from hysteresis module.
        
        Args:
            x: Position tensor
            v: Velocity tensor
        Returns:
            Ghost force tensor (zero if hysteresis disabled)
        """
        if self.hysteresis is not None:
            return self.hysteresis(x, v, topo_id=self.topo_id)
        return torch.zeros_like(v)
    
    def get_friction_coefficient(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None, mu_geo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total friction coefficient mu.
        Hierarchy: 
        1. If mu_geo is provided (and non-zero), use it.
        2. Else, use friction_fallback.
        3. Apply velocity-dependent scaling if enabled.
        """
        if mu_geo is None:
            geo_out = self.geometry(x, v, force=force)
            if isinstance(geo_out, tuple):
                mu_geo = geo_out[1]
            else:
                mu_geo = 0.0
            
        # P0.2 logic: sum both to match oldun's LowRank geometry behavior
        mu_total = self.friction_fallback + mu_geo
        
        if self.velocity_friction_scale > 0:
            # Velocity-dependent scaling (Adaptive Friction)
            v_norm = torch.norm(v, dim=-1, keepdim=True) / (v.shape[-1] ** 0.5 + EPS)
            mu_total = mu_total * (1.0 + self.velocity_friction_scale * v_norm)
            
        return mu_total

    def reset_hysteresis(self):
        """Reset hysteresis state (useful between sequences)."""
        if self.hysteresis is not None:
            self.hysteresis.reset()

    def apply_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Wrap position to manifold domain."""
        if self.is_torus:
            return torch.atan2(torch.sin(x), torch.cos(x))
        return x
