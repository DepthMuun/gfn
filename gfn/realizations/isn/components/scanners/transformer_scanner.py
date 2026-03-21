"""
ISN Transformer Scanner Component — GFN V4
Replaces linear imprinting with causal self-attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...interfaces import BaseScanner

class TransformerScanner(BaseScanner):
    """
    Encoder-style Scanner using standard Multi-Head Attention.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int = 8, 
        n_layers: int = 2,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, token_ids: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l = token_ids.shape
        x = self.embedding(token_ids)
        x = x + self.pos_encoding[:, :l, :]
        
        # Causal mask for the scanner
        mask = nn.Transformer.generate_square_subsequent_mask(l).to(x.device)
        
        impulses = self.transformer(x, mask=mask, is_causal=True)
        return impulses, None # Not truly stateful yet
