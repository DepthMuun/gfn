"""
RK4Integrator — GFN V5
Classic 4th-order Runge-Kutta integrator.
New file (port of legacy euler.py + rk4.py)
"""

import torch
import logging
from typing import Optional, Dict

from ....physics.integrators.base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....registry import register_integrator

logger = logging.getLogger(__name__)


@register_integrator('rk4')
class RK4Integrator(BaseIntegrator):
    """
    Classic 4th-order Runge-Kutta.
    Best accuracy for smooth dynamics; not symplectic.

    k1 = f(x, v)
    k2 = f(x + h/2·v, v + h/2·k1)
    k3 = f(x + h/2·v, v + h/2·k2)
    k4 = f(x + h·v,   v + h·k3)
    x' = x + (h/6)·(v + 2·v + 2·v + v)
    v' = v + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
    """

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt
        h = eff_dt

        curr_x, curr_v = x, v

        for i in range(steps):
            # k1 = f(x, v)
            k1_v = curr_v
            k1_a = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)

            # k2 = f(x + h/2 * k1_v, v + h/2 * k1_a)
            k2_v_val = curr_v + (h / 2.0) * k1_a
            k2_x_val = self._resolve_topology(curr_x + (h / 2.0) * k1_v)
            k2_a = self._get_acceleration(k2_x_val, self._clamp_velocity(k2_v_val), force, dt=eff_dt, **kwargs)
            k2_v = k2_v_val

            # k3 = f(x + h/2 * k2_v, v + h/2 * k2_a)
            k3_v_val = curr_v + (h / 2.0) * k2_a
            k3_x_val = self._resolve_topology(curr_x + (h / 2.0) * k2_v)
            k3_a = self._get_acceleration(k3_x_val, self._clamp_velocity(k3_v_val), force, dt=eff_dt, **kwargs)
            k3_v = k3_v_val

            # k4 = f(x + h * k3_v, v + h * k3_a)
            k4_v_val = curr_v + h * k3_a
            k4_x_val = self._resolve_topology(curr_x + h * k3_v)
            k4_a = self._get_acceleration(k4_x_val, self._clamp_velocity(k4_v_val), force, dt=eff_dt, **kwargs)
            k4_v = k4_v_val

            # Update state using weighted averages
            curr_x = self._resolve_topology(
                curr_x + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            )
            curr_v = self._clamp_velocity(
                curr_v + (h / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
            )

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[RK4] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
