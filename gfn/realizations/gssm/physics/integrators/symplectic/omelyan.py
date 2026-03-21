"""
OmelyanIntegrator — GFN V5
Omelyan PEFRL (Position Extended Forest-Ruth Like) 4th-order symplectic integrator.
Optimized for Hamiltonian systems with significantly better error constants.
"""

import torch
import logging
from typing import Optional, Dict

from ....physics.integrators.base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....registry import register_integrator

logger = logging.getLogger(__name__)

@register_integrator('omelyan')
class OmelyanIntegrator(BaseIntegrator):
    """
    Omelyan PEFRL 4th-order Symplectic Integrator.
    Optimized for energy conservation.
    
    Coefficients (theta, xi, lambda, chi):
    Has approx 100x better error constants than standard Forest-Ruth for some potentials.
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__(physics_engine, config)
        
        # Omelyan PEFRL coefficients
        xi = 0.1786178958448091
        lam = -0.2123418310626054
        chi = -0.06626458266981849
        
        # Drift (c) and Kick (d) weights
        self.c1 = xi
        self.c2 = chi
        self.c3 = 1.0 - 2.0 * (chi + xi)
        self.c4 = chi
        self.c5 = xi
        
        self.d1 = (1.0 - 2.0 * lam) / 2.0
        self.d2 = lam
        self.d3 = lam
        self.d4 = (1.0 - 2.0 * lam) / 2.0

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt

        curr_x, curr_v = x, v

        for i in range(steps):
            # Step 1: Drift-Kick
            curr_x = self._resolve_topology(curr_x + self.c1 * eff_dt * curr_v)
            a1 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d1 * eff_dt * a1)
            
            # Step 2: Drift-Kick
            curr_x = self._resolve_topology(curr_x + self.c2 * eff_dt * curr_v)
            a2 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d2 * eff_dt * a2)
            
            # Step 3: Drift-Kick
            curr_x = self._resolve_topology(curr_x + self.c3 * eff_dt * curr_v)
            a3 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d3 * eff_dt * a3)
            
            # Step 4: Drift-Kick
            curr_x = self._resolve_topology(curr_x + self.c4 * eff_dt * curr_v)
            a4 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d4 * eff_dt * a4)
            
            # Step 5: Final Drift
            curr_x = self._resolve_topology(curr_x + self.c5 * eff_dt * curr_v)

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[Omelyan] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
