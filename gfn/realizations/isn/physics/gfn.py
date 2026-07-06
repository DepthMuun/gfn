"""
Composable ISN physics engine for latent world-flow integration.

This module delegates rollout to pluggable integrator components:

  - ``EulerIntegrator``    for single-state integration
  - ``LeapfrogIntegrator`` for second-order symplectic updates
  - ``YoshidaIntegrator``  for fourth-order symplectic composition

The public forward signature remains stable across integrator choices.
Symplectic variants additionally manage an optional velocity state and expose
``final_velocity`` in the output dictionary.
"""

import math
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..interfaces.base import WorldEngineProtocol
from ..registry import physics

from .integrators import (
    EulerIntegrator,
    EulerState,
    LeapfrogIntegrator,
    SymplecticState,
    YoshidaIntegrator,
    get_integrator,
    list_integrators,
)


_INTEGRATOR_NAMES = list_integrators()


@physics.register("gfn")
class GFNPhysics(nn.Module):
    """
    World Engine that evolves state as a continuous flow in the latent manifold.
    Implements the "Persistent Internal World" pillar.

    Args:
        d_model:        input embedding dimension.
        d_embedding:    latent world dimension.
        integrator:     one of ``"euler"``, ``"leapfrog"``, ``"yoshida"``.
        base_dt:        initial timestep (learnable by default).
        learn_dt:       if True, ``base_dt`` is a learnable parameter.
        learn_initial_v:if True (and integrator is symplectic), initial velocity
                        is a learnable parameter; otherwise a zero buffer.
        velocity_scale: std of the random init for ``initial_v``.
        friction:       optional linear damping (Euler: on state; symplectic:
                        on velocity).
        use_ste:        preserved for compatibility with the STE strategy — falls
                        back to the Euler path because the STE closure operates
                        on a single state tensor.
    """

    def __init__(
        self,
        d_model: int,
        d_embedding: int,
        integrator: str = "euler",
        base_dt: float = 1.0,
        learn_dt: bool = True,
        learn_initial_v: bool = True,
        velocity_scale: float = 0.01,
        friction: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding

        integrator = integrator.lower()
        if integrator not in _INTEGRATOR_NAMES:
            raise ValueError(
                f"integrator must be one of {_INTEGRATOR_NAMES}, got {integrator!r}"
            )
        self.integrator_name = integrator
        self.is_symplectic = integrator in ("leapfrog", "yoshida")

        # ── Learned timestep ──
        if learn_dt:
            self.dt_logit = nn.Parameter(torch.tensor(math.log(base_dt)))
        else:
            self.register_buffer("dt_logit", torch.tensor(math.log(base_dt)))
        self.learn_dt = learn_dt

        # ── Learnable initial velocity for symplectic variants ──
        if self.is_symplectic:
            if learn_initial_v:
                self.initial_v = nn.Parameter(
                    torch.randn(d_embedding) * velocity_scale
                )
            else:
                self.register_buffer("initial_v", torch.zeros(d_embedding))
        else:
            self.initial_v = None

        # ── Integrator component (does NOT hold learnable params) ──
        self.integrator: nn.Module = get_integrator(integrator, friction=friction)

        # ── World-flow learnable parameters ──
        self.drift = nn.Linear(d_embedding, d_embedding)
        self.diffusion = nn.Linear(d_model, d_embedding)
        self.norm = nn.LayerNorm(d_embedding)
        self.use_ste = False

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def base_dt(self) -> float:
        return float(torch.exp(self.dt_logit).detach())

    @property
    def friction(self) -> float:
        return float(self.integrator.friction)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        impulses: torch.Tensor,
        noise_std: float = 0.0,
        world_state: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        b, l, _ = impulses.shape
        device = impulses.device
        dtype = impulses.dtype
        dt = float(torch.exp(self.dt_logit).item())

        drift_w = self.drift.weight
        drift_b = self.drift.bias
        if drift_b is None:
            drift_b = torch.zeros(self.d_embedding, device=device, dtype=dtype)

        # ── STE branch (only valid for Euler; preserved for compatibility) ──
        if self.use_ste and not self.is_symplectic:
            from ..training.strategies.core import StraightThroughEstimator

            state = (
                world_state
                if world_state is not None
                else torch.zeros(b, self.d_embedding, device=device, dtype=dtype)
            )
            emitted, energies = [], []
            for t in range(l):
                f_ext = self.diffusion(impulses[:, t, :])
                state = StraightThroughEstimator._STEFunc.apply(
                    state, f_ext, self.drift
                )
                if noise_std > 0:
                    state = state + torch.randn_like(state) * noise_std
                emitted.append(state.unsqueeze(1))
                energies.append(torch.norm(state, dim=-1, keepdim=True))
            return {
                "emitted_embeddings": self.norm(torch.cat(emitted, dim=1)),
                "energy_trace": torch.cat(energies, dim=1),
                "final_state": state,
            }

        # ── C++ fast-path attempt ──
        f_ext_all = self.diffusion(impulses)
        cuda_out = self._try_cuda_fast_path(
            world_state, velocity_state, f_ext_all,
            drift_w, drift_b, dt, noise_std,
        )
        if cuda_out is not None:
            if self.is_symplectic:
                embs, energies, final_x, final_v = cuda_out
                return {
                    "emitted_embeddings": self.norm(embs),
                    "energy_trace": energies,
                    "final_state": final_x,
                    "final_velocity": final_v,
                }
            embs, energies, final_x = cuda_out
            return {
                "emitted_embeddings": self.norm(embs),
                "energy_trace": energies,
                "final_state": final_x,
            }

        # ── Python fallback (per-component rollout) ──
        if self.is_symplectic:
            state0 = self._make_symplectic_state(b, world_state, velocity_state, device, dtype)
            embs, energies, final_x, final_v = self.integrator.rollout(
                state0, f_ext_all, drift_w, drift_b, dt=dt, noise_std=noise_std
            )
            return {
                "emitted_embeddings": self.norm(embs),
                "energy_trace": energies,
                "final_state": final_x,
                "final_velocity": final_v,
            }
        # Euler
        x0 = (
            world_state
            if world_state is not None
            else torch.zeros(b, self.d_embedding, device=device, dtype=dtype)
        )
        embs, energies, final_x = self.integrator.rollout(
            x0, f_ext_all, drift_w, drift_b, dt=dt, noise_std=noise_std
        )
        return {
            "emitted_embeddings": self.norm(embs),
            "energy_trace": energies,
            "final_state": final_x,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_symplectic_state(
        self,
        batch_size: int,
        world_state: Optional[torch.Tensor],
        velocity_state: Optional[torch.Tensor],
        device,
        dtype,
    ) -> SymplecticState:
        x = (
            world_state
            if world_state is not None
            else torch.zeros(batch_size, self.d_embedding, device=device, dtype=dtype)
        )
        if velocity_state is not None:
            v = velocity_state
        elif self.initial_v is not None:
            v = self.initial_v.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        else:
            v = torch.zeros_like(x)
        return SymplecticState(x=x, v=v)

    def _try_cuda_fast_path(
        self,
        world_state: Optional[torch.Tensor],
        velocity_state: Optional[torch.Tensor],
        f_ext_all: torch.Tensor,
        drift_w: torch.Tensor,
        drift_b: torch.Tensor,
        dt: float,
        noise_std: float,
    ):
        """Try to dispatch to the C++ extension. Returns None on any failure."""
        try:
            csrc_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "csrc", "world_flow")
            )
            if csrc_path not in sys.path:
                sys.path.append(csrc_path)
            import gfn_world_flow  # noqa: F401
        except Exception:
            return None

        try:
            if self.is_symplectic:
                state0 = self._make_symplectic_state(
                    f_ext_all.size(0), world_state, velocity_state,
                    f_ext_all.device, f_ext_all.dtype,
                )
                return gfn_world_flow.world_forward_leapfrog(
                    state0.x, state0.v, f_ext_all, drift_w, drift_b,
                    float(dt), float(self.friction), float(noise_std),
                    self.integrator_name == "yoshida",
                )
            x0 = (
                world_state
                if world_state is not None
                else torch.zeros(f_ext_all.size(0), self.d_embedding,
                                 device=f_ext_all.device, dtype=f_ext_all.dtype)
            )
            return gfn_world_flow.world_forward(
                x0, f_ext_all, drift_w, drift_b, float(noise_std)
            )
        except Exception:
            return None
