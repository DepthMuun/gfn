"""
VerletIntegrator — GFN V5
Velocity Verlet / Störmer-Verlet symplectic integrator.
Migrated from gfn/nn/physics/integrators/symplectic/verlet.py
"""

import torch
import logging
from typing import Optional, Dict

from ....physics.integrators.base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....registry import register_integrator

logger = logging.getLogger(__name__)


@register_integrator('verlet')
class VerletIntegrator(BaseIntegrator):
    """
    Velocity Verlet Symplectic Integrator.

    Algorithm (identical to Leapfrog but written explicitly):
      a0 = F(x)
      x' = x + v·dt + 0.5·a0·dt²
      a1 = F(x')
      v' = v + 0.5·(a0 + a1)·dt
    """

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt

        curr_x, curr_v = x, v

        for i in range(steps):
            # Aceleración inicial
            a0 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            
            # Posición: x' = x + v·dt + 0.5·a0·dt²
            curr_x = self._resolve_topology(curr_x + curr_v * eff_dt + 0.5 * a0 * eff_dt ** 2)
            
            # Nueva aceleración en la posición actualizada
            # Usar velocidad promedio para evaluación más estable
            v_avg = curr_v + 0.5 * a0 * eff_dt
            a1 = self._get_acceleration(curr_x, v_avg, force, dt=eff_dt, **kwargs)
            
            # Velocidad: v' = v + 0.5·(a0 + a1)·dt (promediada)
            curr_v = self._clamp_velocity(curr_v + 0.5 * (a0 + a1) * eff_dt)

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[Verlet] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
