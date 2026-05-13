"""
ISN SSM Emitter Component — GFN V4
O(1) Memory complexity using a simplified State Space Model (SSM) backbone.
"""

import torch
import torch.nn as nn
from ...interfaces import BaseEmitter

class SSMLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.log_a = nn.Parameter(torch.log(torch.linspace(0.9, 0.99, d_model)))
        self.b = nn.Linear(d_model, d_model, bias=False)
        self.c = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, d = u.shape
        a = torch.exp(self.log_a)
        state = torch.zeros(b, d, device=u.device)
        outputs = []
        u_b = self.b(u)
        for t in range(l):
            state = a * state + u_b[:, t, :]
            outputs.append(self.c(state).unsqueeze(1))
        return self.norm(torch.cat(outputs, dim=1))

class SSMEmitter(BaseEmitter):
    """
    Decoder-style Emitter using an SSM backbone.
    """
    def __init__(self, d_embedding: int, vocab_size: int, n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([SSMLayer(d_embedding) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_embedding, vocab_size)

    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        x = emitted_embeddings
        for layer in self.layers:
            x = x + layer(x)
        logits = self.output_proj(x)
        return logits
