"""
BaseIntegrator — GFN V5
Clean base for all symplectic and Runge-Kutta integrators.
Migrated and simplified from gfn/nn/physics/integrators/base.py
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from ...constants import MAX_VELOCITY, TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN, DEFAULT_DT, EPSILON_STANDARD
from ...config.schema import PhysicsConfig
from ...interfaces.physics import PhysicsEngine


class BaseIntegrator(nn.Module):
    """
    Base class for all GFN V5 integrators.

    Provides:
    - Velocity clamping
    - Topology-aware position wrapping
    - Standardized `step()` interface returning Dict[str, Tensor]
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__()
        self.physics_engine = physics_engine
        self.config = config or PhysicsConfig()

        # Expose geometry for backward compat
        if hasattr(physics_engine, 'geometry'):
            self.geometry = physics_engine.geometry
        else:
            self.geometry = physics_engine

        topo_type = self.config.topology.type.lower()
        self.is_torus = (topo_type == TOPOLOGY_TORUS)
        self.velocity_clamp = getattr(self.config.stability, 'velocity_clamp', MAX_VELOCITY)
        self.base_dt = getattr(self.config.stability, 'base_dt', DEFAULT_DT)

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Abstract integration step.
        Args:
            x: Position tensor [B, H, D] or [B, D]
            v: Velocity tensor, same shape as x
            force: External force, same shape as x
            dt:    Step size (overrides config)
        Returns:
            {'x': x_next, 'v': v_next, 'gamma': christoffel_approx (optional)}
        """
        raise NotImplementedError("Subclasses must implement step().")

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _clamp_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Hard clamp to prevent kinetic energy explosion."""
        return torch.clamp(v, -self.velocity_clamp, self.velocity_clamp)

    def _resolve_topology(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrap position to manifold domain.
        Torus: wrap to [-π, π] using atan2(sin, cos).
        Euclidean: identity.
        """
        if self.is_torus:
            return torch.atan2(torch.sin(x), torch.cos(x))
        return x

    def _get_acceleration(self, x: torch.Tensor, v: torch.Tensor,
                          force: Optional[torch.Tensor] = None, dt: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Delegate force computation to the physics engine.
        Returns acceleration tensor (same shape as v).
        """
        res = self.physics_engine.compute_acceleration(x, v, force=force, dt=dt, **kwargs)
        if isinstance(res, tuple):
            return res[0]   # (accel, friction) — return accel only
        return res

    def _resolve_friction_mu(self, x: torch.Tensor, v: torch.Tensor,
                             force: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Unified friction coefficient resolver for all integrators.
        """
        if hasattr(self.physics_engine, 'get_friction_coefficient'):
            return self.physics_engine.get_friction_coefficient(x, v, force=force)
            
        # Fallback for un-refactored engines
        geometry = getattr(self.physics_engine, 'geometry', None)
        mu = None
        if geometry is not None:
            geo_out = geometry(x, v, force=force, **kwargs)
            if isinstance(geo_out, tuple):
                _, mu = geo_out
                
        if mu is None:
            mu = getattr(self.config.stability, 'friction', 0.0)
            
        return mu
