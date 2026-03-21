"""
YoshidaIntegrator — GFN V5
4th-order symplectic integration scheme.
Migrated from gfn/nn/physics/integrators/symplectic/yoshida.py
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


@register_integrator('yoshida')
class YoshidaIntegrator(BaseIntegrator):
    _warned_fallback = False
    """
    Yoshida 4th-order Symplectic Integrator.
    Preserves the symplectic 2-form exactly (to machine precision).

    Coefficients: w1 = 1/(2-∛2), w0 = -∛2/(2-∛2)
    Sequence: c1·x, d1·v, c2·x, d2·v, c3·x, d3·v, c4·x
    """

    def __init__(self, physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        super().__init__(physics_engine, config)

        # 4th-order Yoshida coefficients
        w1 = 1.3512071919596576
        w0 = -1.7024143839193153

        self.c1 = w1 / 2.0
        self.c2 = (w0 + w1) / 2.0
        self.c3 = self.c2
        self.c4 = self.c1

        self.d1 = w1
        self.d2 = w0
        self.d3 = w1

    def step(self, x: torch.Tensor, v: torch.Tensor, force: Optional[torch.Tensor] = None,
             dt: Optional[float] = None, steps: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs `steps` Yoshida steps.
        Returns {'x': x_next, 'v': v_next}.
        """
        eff_dt = (dt if dt is not None else self.base_dt)

        # ── C++ Macro-Kernel Fast Path (O(1) Kernel Launches) ──
        # Check if geometry is LowRank or PaperLowRank and C++ kernel is available
        from ....cuda.ops import yoshida_fused, CUDA_AVAILABLE
        
        geo = getattr(self.physics_engine, 'geometry', None)
        is_low_rank = type(geo).__name__ in ('LowRankRiemannianGeometry', 'PaperLowRankRiemannianGeometry')
        
        if CUDA_AVAILABLE and yoshida_fused is not None and is_low_rank and force is not None and x.is_cuda:
            # Gather params for the C++ ATen inner loop
            U = geo.U             # [H, D, R]
            W = getattr(geo, 'W', geo.U) # W usually same as U in base, but explicit in extensions
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
            
            x_next, v_next = yoshida_fused(
                x, v, U, W, force,
                eff_dt_tens, int(steps), float(clamp_val), float(friction), float(vel_scale), float(v_sat),
                gate_w, gate_b, float(sing_thresh), float(sing_strength),
                bool(trace_norm), bool(is_paper)
            )
                
            return {'x': x_next, 'v': v_next}
            
        # ── Slow PyTorch Python Loop Fallback (O(steps) Kernel Launches) ──
        if not self._warned_fallback:
            logger.warning("[Yoshida] Fused C++ kernel not available or not applicable. Falling back to slow Python loop.")
            YoshidaIntegrator._warned_fallback = True
            
        curr_x, curr_v = x, v

        for i in range(steps):
            # ── Sub-step 1 ──
            curr_x = self._resolve_topology(curr_x + self.c1 * eff_dt * curr_v)
            a1 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d1 * eff_dt * a1)

            # ── Sub-step 2 ──
            curr_x = self._resolve_topology(curr_x + self.c2 * eff_dt * curr_v)
            a2 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d2 * eff_dt * a2)

            # ── Sub-step 3 ──
            curr_x = self._resolve_topology(curr_x + self.c3 * eff_dt * curr_v)
            a3 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt, **kwargs)
            curr_v = self._clamp_velocity(curr_v + self.d3 * eff_dt * a3)

            # ── Final drift ──
            curr_x = self._resolve_topology(curr_x + self.c4 * eff_dt * curr_v)

            # Safety: NaN guard
            if torch.isnan(curr_x).any() or torch.isnan(curr_v).any():
                logger.warning(f"[Yoshida] NaN detected at step {i}/{steps}. Stopping.")
                break

        return {'x': curr_x, 'v': curr_v}
