"""
ISN GFN Physics Engine — Modular V5
================================
O(1) Memory complexity using pure geodetic flow principles.
Restored C++ high-performance support.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from ..interfaces.base import WorldEngineProtocol
from ..registry import physics

@physics.register("gfn")
class GFNPhysics(nn.Module):
    """
    World Engine that evolves state as a continuous flow in the latent manifold.
    Implements the "Persistent Internal World" pillar.
    """
    def __init__(self, d_model: int, d_embedding: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        
        # State transition flow (Symplectic-inspired)
        self.drift = nn.Linear(d_embedding, d_embedding)
        self.diffusion = nn.Linear(d_model, d_embedding)

        self.norm = nn.LayerNorm(d_embedding)
        self.use_ste = False

    def forward(
        self,
        impulses: torch.Tensor,
        noise_std: float = 0.0,
        world_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        b, l, d = impulses.shape
        device = impulses.device

        if world_state is None:
            world_state = torch.zeros(b, self.d_embedding, device=device)

        # ----------------------------------------------------------------------
        # FAST PATH: C++ Extension
        # ----------------------------------------------------------------------
        try:
            # Add extension directory to path if needed
            csrc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/world_flow'))
            if csrc_path not in sys.path:
                sys.path.append(csrc_path)
            
            import gfn_world_flow
            
            # Pre-compute diffusion for all impulses [B, L, D]
            f_ext_all = self.diffusion(impulses)
            
            # Extract weights for the C++ call
            drift_w = self.drift.weight
            drift_b = self.drift.bias
            if drift_b is None:
                drift_b = torch.zeros(self.d_embedding, device=device, dtype=drift_w.dtype)
                
            # Call C++ forward pass
            final_embs, energies, final_state = gfn_world_flow.world_forward(
                world_state, f_ext_all, drift_w, drift_b, float(noise_std)
            )
            
            return {
                'emitted_embeddings': self.norm(final_embs),
                'energy_trace': energies,
                'final_state': final_state
            }
            
        except ImportError:
            # ------------------------------------------------------------------
            # SLOW PATH: Python Fallback
            # ------------------------------------------------------------------
            emitted_embeddings = []
            energy_trace = []
            state = world_state

            for t in range(l):
                if self.use_ste:
                    from ..training.strategies.core import StraightThroughEstimator
                    f_ext = self.diffusion(impulses[:, t, :])
                    state = StraightThroughEstimator._STEFunc.apply(state, f_ext, self.drift)
                else:
                    v_drift = torch.tanh(self.drift(state))
                    f_ext = self.diffusion(impulses[:, t, :])
                    state = state + v_drift + f_ext
                
                if noise_std > 0:
                    state = state + torch.randn_like(state) * noise_std

                emitted_embeddings.append(state.unsqueeze(1))
                energy_trace.append(torch.norm(state, dim=-1, keepdim=True))

            final_embs = self.norm(torch.cat(emitted_embeddings, dim=1))
            energies = torch.cat(energy_trace, dim=1)

            return {
                'emitted_embeddings': final_embs,
                'energy_trace': energies,
                'final_state': state
            }
