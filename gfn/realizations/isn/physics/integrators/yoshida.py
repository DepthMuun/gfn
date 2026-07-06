"""
Fourth-order symplectic composition built on top of the leapfrog kernel.

Dynamics:
    Three leapfrog stages with weights ``w1, w0, w1`` and a final half-kick
    produce a single 4th-order step (local error O(dt^5)).

    w1 = 1 / (2 - 2^(1/3))
    w0 = -2^(1/3) * w1

    for w in (w1, w0, w1):
        v = v + w * dt * a(x, f_ext)
        x = x + w * dt * v
        # optional friction
    v = v + 0.5 * dt * a(x, f_ext)

The integrator inherits ``step``/``rollout`` from :class:`LeapfrogIntegrator`
but exposes itself as a separate swappable component.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .leapfrog import LeapfrogIntegrator, SymplecticState, _acceleration


# Yoshida 4th-order weights (cached as module-level constants)
_CBRT2 = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)
_YOSHIDA_W0 = -_CBRT2 * _YOSHIDA_W1
_YOSHIDA_WEIGHTS = (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1)


class YoshidaIntegrator(LeapfrogIntegrator):
    """
    4th-order Yoshida symplectic integrator.

    Higher accuracy per step than leapfrog (O(dt^5) local error) at roughly
    3x the per-step cost. Recommended when the timestep cannot be reduced
    but extra precision is required.
    """

    name = "yoshida"

    def step(
        self,
        state: SymplecticState,
        f_ext: torch.Tensor,
        drift_w: torch.Tensor,
        drift_b: torch.Tensor,
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> SymplecticState:
        x, v = state.x, state.v
        for w in _YOSHIDA_WEIGHTS:
            a = _acceleration(x, f_ext, drift_w, drift_b)
            v = v + w * dt * a
            x = x + w * dt * v
            if self.friction > 0:
                v = v * max(0.0, 1.0 - self.friction * w * dt)
        # final half-kick (completes the symmetric step)
        a = _acceleration(x, f_ext, drift_w, drift_b)
        v = v + 0.5 * dt * a
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return SymplecticState(x=x, v=v)

    # ── C++ fast-path binding ─────────────────────────────────────────────
    def try_cuda_rollout(self, *args, **kwargs):
        try:
            import gfn_world_flow  # type: ignore

            state0, f_ext_all, drift_w, drift_b = args[:4]
            dt = kwargs.get("dt", 1.0)
            noise_std = kwargs.get("noise_std", 0.0)
            return gfn_world_flow.world_forward_leapfrog(
                state0.x,
                state0.v,
                f_ext_all,
                drift_w,
                drift_b,
                float(dt),
                float(self.friction),
                float(noise_std),
                True,                   # use_yoshida=True
            )
        except Exception:
            return None
