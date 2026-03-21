"""
ISN Linear Scanner Component — GFN V4
Standard sequential imprinting.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...interfaces import BaseScanner

class LinearScanner(BaseScanner):
    """
    Causal/Streaming processor for semantic materialization.
    Pure 1-to-1 functional map.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Base Topologic Pattern
        self.base_imprint = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids_seq: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Shape (batch, seq_len) -> (batch, seq_len, d_model), None
        """
        static_signature = self.base_imprint(token_ids_seq)
        impulses_seq = self.norm(static_signature)
        return impulses_seq, None
