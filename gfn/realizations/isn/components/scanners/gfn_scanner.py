"""
ISN GFN Scanner Component — GFN V4
O(1) Memory complexity using pure geodetic flow principles.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...interfaces import BaseScanner

class GFNScanner(BaseScanner):
    """
    Encoder that treats tokens as impulses and evolves them via a simple flow.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple flow dynamics: learnable shift and rotation
        self.flow_gate = nn.Linear(d_model, d_model)
        self.velocity_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l = token_ids.shape
        impulses = self.embedding(token_ids)
        
        # Evolve state: each token adds to the momentum
        if state is None:
            state = torch.zeros_like(impulses[:, 0, :])
            
        outputs = []
        
        for t in range(l):
            # Impulse updates momentum
            f_ext = impulses[:, t, :]
            # Non-linear flow transformation
            state = torch.tanh(self.flow_gate(state + f_ext))
            outputs.append(state.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), state
