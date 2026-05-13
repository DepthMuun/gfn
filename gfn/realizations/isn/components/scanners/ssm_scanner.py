"""
ISN SSM Scanner Component — GFN V4
O(1) Memory complexity using a simplified State Space Model (SSM) backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from ...interfaces import BaseScanner

class SSMLayer(nn.Module):
    """
    Simplified Linear Recurrent Unit (LRU) / SSM Layer.
    x_t = A * x_{t-1} + B * u_t
    y_t = C * x_t
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # A matrix (diagonal approximation for efficiency)
        self.log_a = nn.Parameter(torch.log(torch.linspace(0.9, 0.999, d_model)))
        # B matrix
        self.b = nn.Linear(d_model, d_model, bias=False)
        # C matrix
        self.c = nn.Linear(d_model, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (batch, seq_len, d_model)
            state: (batch, d_model)
        Returns:
            y: (batch, seq_len, d_model)
            state: (batch, d_model)
        """
        b_size, seq_len, d_model = u.shape
        a = torch.exp(self.log_a)
        
        if state is None:
            state = torch.zeros(b_size, d_model, device=u.device)
            
        outputs = []
        u_b = self.b(u)
        
        for t in range(seq_len):
            state = a * state + u_b[:, t, :]
            y_t = self.c(state)
            outputs.append(y_t.unsqueeze(1))
            
        y = torch.cat(outputs, dim=1)
        return self.norm(y), state

class SSMScanner(BaseScanner):
    """
    Encoder-style Scanner using an SSM backbone.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SSMLayer(d_model) for _ in range(n_layers)])

    def forward(self, token_ids: torch.Tensor, state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.embedding(token_ids)
        new_states = []
        for i, layer in enumerate(self.layers):
            s = state[i] if state is not None else None
            out, s_new = layer(x, state=s)
            x = x + out
            new_states.append(s_new)
        return x, new_states
