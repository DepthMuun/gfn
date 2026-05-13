"""
LeapfrogIntegrator — GFN V5
Störmer-Verlet (Leapfrog) 2nd-order symplectic integrator.
Migrated from gfn/nn/physics/integrators/symplectic/leapfrog.py
"""

import torch
import logging
from typing import Optional, Dict

from ..base import BaseIntegrator
from ....interfaces.physics import PhysicsEngine
from ....config.schema import PhysicsConfig
from ....constants import EPSILON_STANDARD
from ....registry import register_integrator

logger = logging.getLogger(__name__)


@register_integrator('leapfrog')
class LeapfrogIntegrator(BaseIntegrator):
    _warned_fallback = False
    """
    Leapfrog / Störmer-Verlet 2nd-order Symplectic Integrator.

    Algorithm:
      v_half = v + 0.5·dt·a(x)
      x_next = x + dt·v_half
      v_next = v_half + 0.5·dt·a(x_next)

    Efficient: requires only 1 force eval per step (vs 3 for Yoshida).
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__(physics_engine, config)


    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        eff_dt = dt if dt is not None else self.base_dt
        
        # ── C++ Macro-Kernel Fast Path (O(1) Kernel Launches) ──
        from ....cuda.ops import leapfrog_fused, CUDA_AVAILABLE
        
        geo = getattr(self.physics_engine, 'geometry', None)
        is_low_rank = type(geo).__name__ in ('LowRankRiemannianGeometry', 'PaperLowRankRiemannianGeometry')
        
        if CUDA_AVAILABLE and leapfrog_fused is not None and is_low_rank and force is not None and x.is_cuda:
            U = geo.U
            W = getattr(geo, 'W', geo.U)
            clamp_val = geo.clamp_val
            friction = self.config.stability.friction
            vel_scale = self.config.stability.velocity_friction_scale
            trace_norm = geo.enable_trace_normalization
            is_paper = type(geo).__name__ == 'PaperLowRankRiemannianGeometry'
            
            # C++ macro-kernel handles 3D tensors [B, H, D] or 2D [B, D] natively via ATen broadcasting.
            # No flattening needed here as it breaks the head-wise weights lookup in C++ matmul.
            if not isinstance(eff_dt, torch.Tensor):
                eff_dt_tens = torch.tensor([eff_dt], dtype=x.dtype, device=x.device)
            else:
                eff_dt_tens = eff_dt
            
            v_sat = self.config.stability.velocity_saturation
            
            # AI Parameters
            gate_w = getattr(geo.friction_gate, 'gate_w', torch.empty(0, device=x.device))
            gate_b = getattr(geo.friction_gate, 'gate_b', torch.empty(0, device=x.device))
            
            sing_cfg = self.config.singularities
            sing_thresh = sing_cfg.threshold if sing_cfg.enabled else 0.0
            sing_strength = sing_cfg.strength if sing_cfg.enabled else 0.0
            
            x_next, v_next = leapfrog_fused(
                x, v, U, W, force,
                eff_dt_tens, int(steps), float(clamp_val), float(friction), float(vel_scale), float(v_sat),
                gate_w, gate_b, float(sing_thresh), float(sing_strength),
                bool(trace_norm), bool(is_paper)
            )
            
            return {'x': x_next, 'v': v_next}

        # ── Slow PyTorch Python Loop Fallback (O(steps) Kernel Launches) ──
        if not self._warned_fallback:
            logger.warning("[Leapfrog] Fused C++ kernel not available or not applicable. Falling back to slow Python loop.")
            LeapfrogIntegrator._warned_fallback = True
            
        curr_x, curr_v = x, v

        for i in range(steps):
            # Half-kick velocity
            mu1 = self._resolve_friction_mu(curr_x, curr_v, force=force, **kwargs)
            a1 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            a1_nf = a1 + mu1 * curr_v
            v_half = (curr_v + 0.5 * eff_dt * a1_nf) / (1.0 + 0.5 * eff_dt * mu1 + EPSILON_STANDARD)
            v_half = self._clamp_velocity(v_half)

            # Full drift position
            curr_x = self._resolve_topology(curr_x + eff_dt * v_half)

            # Re-evaluate acceleration at new position with velocity average
            # Usamos v_half que es la velocidad promedio en el intervalo
            mu2 = self._resolve_friction_mu(curr_x, v_half, force=force, **kwargs)
            a2 = self._get_acceleration(curr_x, v_half, force, dt=eff_dt, **kwargs)
            a2_nf = a2 + mu2 * v_half

            # Final half-kick con aceleración promediada (más estable)
            a_avg = (a1_nf + a2_nf) / 2
            mu_avg = (mu1 + mu2) / 2
            curr_v = (curr_v + eff_dt * a_avg) / (1.0 + eff_dt * mu_avg + EPSILON_STANDARD)
            curr_v = self._clamp_velocity(curr_v)

            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[Leapfrog] NaN at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
