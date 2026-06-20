"""
Leapfrog Integrator Component — ISN v2.7.3
==========================================
Störmer-Verlet (2nd-order symplectic) integration of the world flow.

Dynamics (split Hamiltonian form with ``H(x, v) = ½‖v‖² + V(x)`` and
``a(x, f_ext) = tanh(W x + b) + f_ext``):

    v_half = v + 0.5 * dt * a(x, f_ext)
    x_new  = x + dt * v_half
    v_new  = v_half + 0.5 * dt * a(x_new, f_ext)

The integrator conserves a modified energy up to ``O(dt²)`` per step and is
unconditionally stable for separable Hamiltonians. A linear friction on
velocity is applied after each step.

The companion C++ fast path ``world_forward_leapfrog`` lives in
``gfn/realizations/isn/csrc/world_flow/world_flow.cpp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SymplecticState:
    """Position-velocity pair used by symplectic integrators."""

    x: torch.Tensor  # [B, D]
    v: torch.Tensor  # [B, D]


def _acceleration(
    x: torch.Tensor,
    f_ext: torch.Tensor,
    drift_w: torch.Tensor,
    drift_b: torch.Tensor,
) -> torch.Tensor:
    """dv/dt = tanh(W x + b) + f_ext."""
    return torch.tanh(F.linear(x, drift_w, drift_b)) + f_ext


class LeapfrogIntegrator(nn.Module):
    """
    Störmer-Verlet (Leapfrog) 2nd-order symplectic integrator.

    Args:
        friction: optional linear damping coefficient on velocity (>= 0).
    """

    name = "leapfrog"

    def __init__(self, friction: float = 0.0):
        super().__init__()
        self.friction = float(friction)

    @staticmethod
    def initial_state(
        batch_size: int,
        d_embedding: int,
        device,
        dtype,
        initial_v: Optional[torch.Tensor] = None,    # [D] or [B, D]
    ) -> SymplecticState:
        x = torch.zeros(batch_size, d_embedding, device=device, dtype=dtype)
        if initial_v is None:
            v = torch.zeros_like(x)
        elif initial_v.dim() == 1:
            v = initial_v.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        else:
            v = initial_v.to(device=device, dtype=dtype)
        return SymplecticState(x=x, v=v)

    def step(
        self,
        state: SymplecticState,
        f_ext: torch.Tensor,                 # [B, D]
        drift_w: torch.Tensor,
        drift_b: torch.Tensor,
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> SymplecticState:
        a = _acceleration(state.x, f_ext, drift_w, drift_b)
        v_half = state.v + 0.5 * dt * a
        x_new = state.x + dt * v_half
        a_new = _acceleration(x_new, f_ext, drift_w, drift_b)
        v_new = v_half + 0.5 * dt * a_new
        if self.friction > 0:
            v_new = v_new * max(0.0, 1.0 - self.friction * dt)
        if noise_std > 0:
            x_new = x_new + torch.randn_like(x_new) * noise_std
        return SymplecticState(x=x_new, v=v_new)

    def rollout(
        self,
        state0: SymplecticState,
        f_ext_all: torch.Tensor,              # [B, L, D]
        drift_w: torch.Tensor,
        drift_b: torch.Tensor,
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            embs:     [B, L, D]
            energies: [B, L, 1]
            final_x:  [B, D]
            final_v:  [B, D]
        """
        state = SymplecticState(x=state0.x.clone(), v=state0.v.clone())
        L = f_ext_all.size(1)
        embs, energies = [], []
        for t in range(L):
            state = self.step(state, f_ext_all[:, t, :], drift_w, drift_b,
                              dt=dt, noise_std=noise_std)
            embs.append(state.x.unsqueeze(1))
            energies.append(torch.norm(state.x, dim=-1, keepdim=True).unsqueeze(1))  # [B, 1, 1]
        return (
            torch.cat(embs, dim=1),
            torch.cat(energies, dim=1),
            state.x,
            state.v,
        )

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
                False,                  # use_yoshida=False
            )
        except Exception:
            return None