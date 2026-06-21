"""
Scanner that maps token IDs to impulse vectors through learned embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ...interfaces.base import ScannerProtocol
from ...registry import scanners

@scanners.register("gfn")
class GFNScanner(nn.Module):
    """
    Default scanner that embeds token IDs into the latent state pipeline.
    """
    def __init__(self, vocab_size: int, d_model: int, d_embedding: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self, 
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # b, l = token_ids.shape
        x = self.embedding(token_ids)
        impulses = torch.tanh(self.projection(x))
        return impulses, None
