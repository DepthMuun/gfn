"""
Transformer-based scanner using causal self-attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...interfaces.base import ScannerProtocol
from ...registry import scanners

@scanners.register("transformer")
class TransformerScanner(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int = 8, 
        n_layers: int = 2,
        max_seq_len: int = 8192,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
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
        # Handle positional encoding length
        if l <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :l, :]
        
        mask = nn.Transformer.generate_square_subsequent_mask(l).to(x.device)
        impulses = self.transformer(x, mask=mask, is_causal=True)
        return impulses, None
