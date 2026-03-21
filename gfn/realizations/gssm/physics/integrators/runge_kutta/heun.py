"""
HeunIntegrator — GFN V5
Heun (explicit trapezoidal) 2nd-order Runge-Kutta integrator.
Migrated from gfn/nn/physics/integrators/runge_kutta/heun.py
"""

import torch
import logging
from typing import Optional, Dict

from ....physics.integrators.base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....registry import register_integrator

logger = logging.getLogger(__name__)


@register_integrator('heun')
class HeunIntegrator(BaseIntegrator):
    """
    Heun's method (explicit trapezoidal rule), 2nd-order RK.

    Algorithm:
      k1 = f(x, v)
      x̃ = x + dt·v,  ṽ = v + dt·k1
      k2 = f(x̃, ṽ)
      x' = x + 0.5·dt·(v + ṽ)
      v' = v + 0.5·dt·(k1 + k2)

    Not strictly symplectic, but good for non-conservative systems.
    """

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt

        curr_x, curr_v = x, v

        for i in range(steps):
            # k1: acceleration at current state
            k1_a = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            k1_v = curr_v

            # Euler predictor
            x_pred = self._resolve_topology(curr_x + eff_dt * k1_v)
            v_pred = self._clamp_velocity(curr_v + eff_dt * k1_a)

            # k2: acceleration at predicted state
            k2_a = self._get_acceleration(x_pred, v_pred, force, dt=eff_dt, **kwargs)
            k2_v = v_pred

            # Trapezoidal corrector
            curr_x = self._resolve_topology(curr_x + 0.5 * eff_dt * (k1_v + k2_v))
            curr_v = self._clamp_velocity(curr_v + 0.5 * eff_dt * (k1_a + k2_a))

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[Heun] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
