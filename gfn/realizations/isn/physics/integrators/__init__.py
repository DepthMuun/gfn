"""
ISN Physics Integrators — Modular v2.7.3
=====================================
Swappable integrator components for the world flow.

Available integrators
---------------------
- ``EulerIntegrator``  : forward-Euler (backward-compatible default).
- ``LeapfrogIntegrator``: Störmer-Verlet 2nd-order symplectic.
- ``YoshidaIntegrator`` : 4th-order Yoshida composition on top of leapfrog.

All integrators share the same interface:

    integrator.rollout(state0, f_ext_all, drift_w, drift_b, dt, noise_std)
        -> (embs, energies, final_x[, final_v])

and may be obtained through :func:`get_integrator` by name.
"""

from typing import Dict, Type

import torch.nn as nn

from .euler import EulerIntegrator, EulerState
from .leapfrog import LeapfrogIntegrator, SymplecticState
from .yoshida import YoshidaIntegrator


_REGISTRY: Dict[str, Type[nn.Module]] = {
    "euler":    EulerIntegrator,
    "leapfrog": LeapfrogIntegrator,
    "yoshida":  YoshidaIntegrator,
}


def get_integrator(name: str, **kwargs) -> nn.Module:
    """Return an instance of the requested integrator (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown integrator {name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_integrators():
    return sorted(_REGISTRY.keys())


__all__ = [
    "EulerIntegrator",
    "EulerState",
    "LeapfrogIntegrator",
    "SymplecticState",
    "YoshidaIntegrator",
    "get_integrator",
    "list_integrators",
]