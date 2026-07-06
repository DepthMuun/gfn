"""
Forward-Euler integration of the ISN world flow.

Dynamics:
    x_{t+1} = x_t + tanh(W x_t + b) + f_ext_t (+ noise)

This integrator uses a single state tensor and does not maintain a separate
velocity state. Its tensor interface matches the earlier Euler-based physics
path so existing callers can continue to use it unchanged.

The companion C++ fast path ``world_forward_euler`` lives in
``gfn/realizations/isn/csrc/world_flow/world_flow.cpp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EulerState:
    """Position-only state used by the Euler integrator."""

    x: torch.Tensor  # [B, D]


class EulerIntegrator(nn.Module):
    """
    Forward-Euler integrator for the ISN world flow.

    Args:
        friction:  optional linear damping applied to ``x`` after each step
                   (default 0.0 preserves the original Euler behaviour).
    """

    name = "euler"

    def __init__(self, friction: float = 0.0):
        super().__init__()
        self.friction = float(friction)

    @staticmethod
    def initial_state(batch_size: int, d_embedding: int, device, dtype) -> "EulerState":
        return EulerState(x=torch.zeros(batch_size, d_embedding, device=device, dtype=dtype))

    def step(
        self,
        state: EulerState,
        f_ext: torch.Tensor,                 # [B, D]
        drift_w: torch.Tensor,                # [D, D]
        drift_b: torch.Tensor,                # [D]
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> EulerState:
        drift = torch.tanh(torch.nn.functional.linear(state.x, drift_w, drift_b))
        x = state.x + dt * drift + dt * f_ext
        if self.friction > 0:
            x = x * max(0.0, 1.0 - self.friction * dt)
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return EulerState(x=x)

    def rollout(
        self,
        x0: torch.Tensor,                     # [B, D]
        f_ext_all: torch.Tensor,              # [B, L, D]
        drift_w: torch.Tensor,
        drift_b: torch.Tensor,
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            embs:      [B, L, D]
            energies:  [B, L, 1]   ||x_t|| per step
            final_x:   [B, D]
        """
        b, L, _ = f_ext_all.shape
        x = x0
        embs = []
        energies = []
        for t in range(L):
            x = self.step(EulerState(x=x), f_ext_all[:, t, :], drift_w, drift_b,
                          dt=dt, noise_std=noise_std).x
            embs.append(x.unsqueeze(1))
            energies.append(torch.norm(x, dim=-1, keepdim=True).unsqueeze(1))   # [B, 1, 1]
        return (
            torch.cat(embs, dim=1),                                # [B, L, D]
            torch.cat(energies, dim=1),                            # [B, L, 1]
            x,                                                     # [B, D]
        )

    # ── C++ fast-path binding ─────────────────────────────────────────────
    def try_cuda_rollout(self, *args, **kwargs):
        """Optional C++ binding (returns None when the extension is missing)."""
        try:
            import gfn_world_flow  # type: ignore

            x0, f_ext_all, drift_w, drift_b, dt, noise_std = (
                args[0], args[1], args[2], args[3],
                kwargs.get("dt", 1.0),
                kwargs.get("noise_std", 0.0),
            )
            return gfn_world_flow.world_forward(
                x0, f_ext_all, drift_w, drift_b, float(noise_std)
            )
        except Exception:
            return None
