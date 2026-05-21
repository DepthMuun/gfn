"""
ISN GFN Emitter — Modular V5
==========================
Physical states -> Prediction logits.
"""

import torch
import torch.nn as nn
from ...interfaces.base import EmitterProtocol
from ...registry import emitters

@emitters.register("gfn")
class GFNEmitter(nn.Module):
    """
    Standard GFN Emitter that reconstructs tokens from world embeddings.
    """
    def __init__(self, d_embedding: int, vocab_size: int, **kwargs):
        super().__init__()
        self.emission = nn.Linear(d_embedding, vocab_size, bias=False)

    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        return self.emission(emitted_embeddings)
