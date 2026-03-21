"""
AdaptiveIntegrator — GFN V5
Dyanmic time-stepping solver based on local geometry curvature and acceleration.
Wraps a base integrator and modulates dt per step.
"""

import torch
import logging
from typing import Optional, Dict, Any

from ...physics.integrators.base import BaseIntegrator
from ...interfaces.physics import PhysicsEngine
from ...config.schema import PhysicsConfig
from ...registry import register_integrator, INTEGRATOR_REGISTRY

logger = logging.getLogger(__name__)

@register_integrator('adaptive')
class AdaptiveIntegrator(BaseIntegrator):
    """
    Adaptive Time-Step Integrator.
    Automatically scales dt based on:
    dt_eff = base_dt / (1.0 + alpha * ||accel||)
    
    Args:
        base_integrator_type: The underlying solver ('verlet', 'rk4', etc.)
        alpha: Sensitivity to acceleration/curvature (default: 0.1)
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__(physics_engine, config)
        
        # Adaptive params from config or defaults
        self.alpha = getattr(self.config.stability, 'adaptive_alpha', 0.1)
        self.dt_min = getattr(self.config.stability, 'dt_min', 0.001)
        
        # Base integrator for the actual steps
        base_type = getattr(self.config.stability, 'base_solver', 'verlet')
        if base_type == 'adaptive': # Prevent recursion
            base_type = 'verlet'
            
        base_cls = INTEGRATOR_REGISTRY.get(base_type)
        self.base_solver = base_cls(physics_engine, config)

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        
        curr_x, curr_v = x, v
        base_dt = dt if dt is not None else self.base_dt
        
        for i in range(steps):
            # 1. Estimate local curvature/acceleration
            accel = self._get_acceleration(curr_x, curr_v, force, dt=base_dt, **kwargs)
            acc_norm = torch.norm(accel, dim=-1, keepdim=True)
            
            # 2. Compute adaptive dt
            # Scale down dt where acceleration is high
            dt_eff = base_dt / (1.0 + self.alpha * acc_norm)
            dt_eff = torch.clamp(dt_eff, min=self.dt_min, max=base_dt)
            
            # Mean dt for the batch (simpler for vectorized steps)
            dt_val = dt_eff.mean().item()
            
            # 3. Step using base solver
            res = self.base_solver.step(curr_x, curr_v, force=force, dt=dt_val, steps=1, **kwargs)
            curr_x, curr_v = res['x'], res['v']
            
            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                break
                
        return {'x': curr_x, 'v': curr_v}
