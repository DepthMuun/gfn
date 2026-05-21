"""
ISN Adjoint Strategy — Modular V5
=============================
Implements O(1) memory complexity using the Adjoint State Method.
Treats the GFN physics flow as a continuous ODE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from .core import BaseStrategy
from ...registry import strategies

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    odeint = None

class ISN_ODEFunc(nn.Module):
    """
    Wraps ISN Physics into a continuous-time ODE function.
    """
    def __init__(self, physics: nn.Module, impulses: torch.Tensor):
        super().__init__()
        self.physics = physics
        self.impulses = impulses
        self.n_steps = impulses.shape[1]

    def forward(self, t, s_tuple):
        """
        ds/dt = f(s) + g(u)
        s_tuple: (state_tensor,)
        """
        s = s_tuple[0]
        # Piece-wise index mapping
        idx = torch.clamp(t.long(), 0, self.n_steps - 1)
        u_t = self.impulses[:, idx]
        
        # GFN Physics: s_dot = drift(s) + diffusion(u)
        v_drift = torch.tanh(self.physics.drift(s))
        f_ext = self.physics.diffusion(u_t)
        
        return (v_drift + f_ext,)

@strategies.register("adjoint")
class AdjointStrategy(BaseStrategy):
    """
    Backpropagation Strategy using the Adjoint State Method.
    Provides O(1) memory complexity for the physics world.
    """
    def __init__(self, method: str = 'euler', rtol: float = 1e-3, atol: float = 1e-3, **kwargs):
        super().__init__()
        if odeint is None:
            raise ImportError("AdjointStrategy requires 'torchdiffeq'. Please install it: pip install torchdiffeq")
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def compute_loss(
        self,
        targets: torch.Tensor,
        model: nn.Module,
        **outputs
    ) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {'loss': loss}

    def prepare_model(self, model: nn.Module):
        """
        Wraps the model's world engine to use the Adjoint method during forward.
        """
        def adjoint_forward(impulses: torch.Tensor, **kwargs):
            b, l, d = impulses.shape
            device = impulses.device
            
            # Robust s0 initialization
            s0 = kwargs.get('world_state')
            if s0 is None:
                s0 = torch.zeros(b, model.world.d_embedding, device=device)
            
            t = torch.linspace(0, l, l + 1).to(device).float()
            ode_func = ISN_ODEFunc(model.world, impulses)
            
            # Use tuple-based state packing for torchdiffeq compatibility
            S_tuple = odeint(
                ode_func,
                (s0,),
                t,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            
            S = S_tuple[0] # [L+1, B, D]
            
            # Pack back into standard world_output format
            final_embs = S[1:].transpose(0, 1) # [B, L, D]
            final_embs = model.world.norm(final_embs)
            
            return {
                'emitted_embeddings': final_embs,
                'final_state': S[-1],
                'energy_trace': torch.norm(final_embs, dim=-1, keepdim=True)
            }
            
        model.world.forward = adjoint_forward
        print("✓ Adjoint Gradient Method enabled (O(1) World Memory)")
