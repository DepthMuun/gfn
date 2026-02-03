"""
Implicit Neural Field Embedding
================================

Maps Token IDs → Learnable Coordinates → Vector Space via SIREN.
"""

import torch
import torch.nn as nn
from .siren import SineLayer
from ..constants import EMBEDDING_SCALE


class ImplicitEmbedding(nn.Module):
    """
    Implicit Neural Field Embedding.
    
    Maps Token IDs → Learnable Coordinates → Vector Space via SIREN.
    
    Instead of storing a full embedding table E[vocab_size, emb_dim],
    we store low-rank coordinates C[vocab_size, coord_dim] and learn
    a continuous function f: R^coord_dim → R^emb_dim.
    
    Parameter Reduction:
        Standard: vocab_size * emb_dim
        Implicit: vocab_size * coord_dim + SIREN_params
        
        Example: 10k vocab, 256 dim
        - Standard: 10k * 256 = 2.56M params
        - Implicit: 10k * 16 + ~50k = 210k params (~12x reduction)
    
    Args:
        vocab_size: Number of tokens (for coordinate table size)
        emb_dim: Output embedding dimension
        coord_dim: Dimension of coordinate space (default: 16)
        hidden_dim: Hidden dimension of SIREN network (default: 64)
        layers: Number of SIREN hidden layers (default: 2)
    
    Examples:
        >>> # Basic usage
        >>> embedding = ImplicitEmbedding(vocab_size=10000, emb_dim=256)
        >>> input_ids = torch.randint(0, 10000, (2, 10))
        >>> output = embedding(input_ids)  # [2, 10, 256]
        
        >>> # Custom configuration
        >>> embedding = ImplicitEmbedding(
        ...     vocab_size=10000,
        ...     emb_dim=256,
        ...     coord_dim=32,
        ...     hidden_dim=128,
        ...     layers=3
        ... )
    """
    
    def __init__(self, vocab_size, emb_dim, coord_dim=16, hidden_dim=64, layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.coord_dim = coord_dim
        
        # 1. Coordinate Map (Low-Rank)
        # Much smaller than a full embedding table.
        # e.g. 10k tokens * 16 dim = 160k params (vs 10k * 256 = 2.5M params)
        self.coords = nn.Embedding(vocab_size, coord_dim)
        
        # Init coordinates uniformly to spread them out
        nn.init.uniform_(self.coords.weight, -1.0, 1.0)
        
        # 2. Continuous Function f(c) -> v
        # SIREN MLP
        net = []
        
        # Input Layer
        net.append(SineLayer(coord_dim, hidden_dim, is_first=True, omega_0=30.0))
        
        # Hidden Layers
        for _ in range(layers):
            net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=30.0))
            
        # Output Linear Projection (to match emb_dim magnitude correctly)
        # We don't use Sine on output to allow unbounded range if needed,
        # but typically embeddings are loose.
        self.net = nn.Sequential(*net)
        
        # Final projection to exact dimension
        self.out_proj = nn.Linear(hidden_dim, emb_dim)
        
        # Init final linear to be reasonable magnitude (match standard embedding scale)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            
    def forward(self, input_ids):
        """
        Forward pass through implicit embedding.
        
        Args:
            input_ids: [batch, seq_len] - Token indices
            
        Returns:
            embeddings: [batch, seq_len, emb_dim] - Embedded vectors
        """
        # 1. Lookup Coordinates
        # c: [batch, seq, coord_dim]
        c = self.coords(input_ids)
        
        # 2. Evaluate Field
        # x: [batch, seq, hidden]
        x = self.net(c)
        
        # 3. Project
        out = self.out_proj(x)
        
        return out * EMBEDDING_SCALE  # Moderated boost for better gradients
