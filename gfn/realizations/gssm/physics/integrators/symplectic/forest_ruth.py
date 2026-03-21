"""
ForestRuthIntegrator — GFN V5
Forest-Ruth 4th-order symplectic integrator.
Migrated from gfn/nn/physics/integrators/symplectic/forest_ruth.py
"""

import torch
import logging
from typing import Optional, Dict

from ....physics.integrators.base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....registry import register_integrator

logger = logging.getLogger(__name__)


@register_integrator('forest_ruth')
class ForestRuthIntegrator(BaseIntegrator):
    """
    Forest-Ruth 4th-order Symplectic Integrator.
    Same coefficient structure as Yoshida but with explicit Forest-Ruth theta.

    theta = 1/(2 - ∛2)
    Sequence: c1·x, d1·v, c2·x, d2·v, c3·x, d3·v, c4·x
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__(physics_engine, config)

        theta = 1.3512071919596576  # 1/(2 - ∛2)

        self.c1 = theta / 2.0
        self.c2 = (1.0 - theta) / 2.0
        self.c3 = self.c2
        self.c4 = self.c1

        self.d1 = theta
        self.d2 = 1.0 - 2.0 * theta
        self.d3 = theta

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt

        curr_x, curr_v = x, v

        for i in range(steps):
            # Sub-step 1
            curr_x = self._resolve_topology(curr_x + self.c1 * eff_dt * curr_v)
            a1 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d1 * eff_dt * a1)

            # Sub-step 2
            curr_x = self._resolve_topology(curr_x + self.c2 * eff_dt * curr_v)
            a2 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d2 * eff_dt * a2)

            # Sub-step 3
            curr_x = self._resolve_topology(curr_x + self.c3 * eff_dt * curr_v)
            a3 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d3 * eff_dt * a3)

            # Final drift
            curr_x = self._resolve_topology(curr_x + self.c4 * eff_dt * curr_v)

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[ForestRuth] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
