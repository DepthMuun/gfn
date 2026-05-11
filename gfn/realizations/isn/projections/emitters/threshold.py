"""
ISN Threshold Emitter — Modular V5
================================
Emitter with non-linear threshold activation.
"""

import torch
import torch.nn as nn
from ...interfaces.base import EmitterProtocol
from ...registry import emitters

@emitters.register("threshold")
class ThresholdEmitter(nn.Module):
    """
    Emitter that applies a tanh threshold before linear projection.
    """
    def __init__(self, d_embedding: int, vocab_size: int, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_embedding, vocab_size)

    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.tanh(emitted_embeddings))
