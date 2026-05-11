"""gfn/physics/dynamics/gated.py — GatedDynamics: learnable gate between current and proposal."""
import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class GatedDynamics(BaseDynamics):
    """
    Gated Dynamics: learnable gate that interpolates between current and proposal.

    g = sigmoid(W_g * [current; proposal])
    state_next = norm(g * proposal + (1-g) * current)
    
    More expressive than MixDynamics because the gate depends on the current state.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN, **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.gate[0].bias)
        nn.init.xavier_uniform_(self.gate[0].weight, gain=0.5)

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        gate_input = torch.cat([current_state, absolute_proposal], dim=-1)
        g = self.gate(gate_input)
        mixed = g * absolute_proposal + (1.0 - g) * current_state
        return self._apply_norm(mixed, context_x=context_x)
