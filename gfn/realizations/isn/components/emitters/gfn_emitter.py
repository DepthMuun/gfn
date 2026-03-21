"""
ISN GFN Emitter Component — GFN V4
O(1) Memory complexity using pure geodetic flow principles.
"""

import torch
import torch.nn as nn
from ...interfaces import BaseEmitter

class GFNEmitter(BaseEmitter):
    """
    Decoder that projects world states through a final geodetic flow.
    """
    def __init__(self, d_embedding: int, vocab_size: int):
        super().__init__()
        self.flow_core = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.Tanh(),
            nn.Linear(d_embedding, d_embedding)
        )
        self.output_proj = nn.Linear(d_embedding, vocab_size)

    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        # Final flow refinement
        refined = self.flow_core(emitted_embeddings)
        logits = self.output_proj(refined)
        return logits
