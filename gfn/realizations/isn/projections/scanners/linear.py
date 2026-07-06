"""
Scanner that maps tokens to impulses through a learned embedding table.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ...interfaces.base import ScannerProtocol
from ...registry import scanners

@scanners.register("linear")
class LinearScanner(nn.Module):
    """
    Direct linear projection scanner.
    """
    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(
        self, 
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        impulses = self.embedding(token_ids)
        return impulses, None
