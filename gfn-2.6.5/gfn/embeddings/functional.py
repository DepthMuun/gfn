"""
Functional Embedding
====================

Pure functional embedding with O(1) memory (doesn't scale with vocab size).
"""

import torch
import torch.nn as nn
import numpy as np
from .siren import SineLayer
from ..constants import SIREN_OMEGA_0


class FunctionalEmbedding(nn.Module):
    """
    Pure Functional Embedding (Zero-Lookup).
    
    Maps Index → Coordinate → SIREN → Vector.
    
    O(1) Memory: Parameters do NOT scale with Vocab Size.
    
    Modes:
        - 'sinusoidal': High-freq hash (good for input uniqueness, bad for readout)
        - 'binary': Bitwise representation (good for learning/readout)
        - 'linear': Direct coordinate mapping (essential for parity/arithmetic)
    
    Args:
        vocab_size: Maximum vocabulary size (for reference, not used in params)
        emb_dim: Output embedding dimension
        coord_dim: Coordinate dimension (default: 16)
        hidden_dim: Hidden dimension of SIREN (default: 64)
        layers: Number of SIREN layers (default: 2)
        mode: Coordinate mode ('sinusoidal', 'binary', 'linear')
        impulse_scale: Learnable scale factor (default: 1.0)
        omega_0: SIREN frequency parameter
    
    Examples:
        >>> # Binary mode (good for learning)
        >>> embedding = FunctionalEmbedding(
        ...     vocab_size=10000,
        ...     emb_dim=256,
        ...     mode='binary'
        ... )
        
        >>> # Linear mode (for arithmetic tasks)
        >>> embedding = FunctionalEmbedding(
        ...     vocab_size=10000,
        ...     emb_dim=256,
        ...     mode='linear'
        ... )
        
        >>> # Sinusoidal mode (for hashing)
        >>> embedding = FunctionalEmbedding(
        ...     vocab_size=10000,
        ...     emb_dim=256,
        ...     mode='sinusoidal'
        ... )
    """
    
    def __init__(self, vocab_size, emb_dim, coord_dim=16, hidden_dim=64, layers=2, 
                 mode='binary', impulse_scale=1.0, omega_0=SIREN_OMEGA_0):
        super().__init__()
        self.mode = mode
        self.coord_dim = coord_dim
        self.omega_0 = omega_0
        self.impulse_scale = nn.Parameter(torch.tensor(impulse_scale), requires_grad=True)
            
        if self.mode == 'linear':
            self.net = nn.Identity()
            self.out_proj = nn.Linear(self.coord_dim, emb_dim)
            # Level 32: Dense Broadcast Initialization
            # When supervising all manifold dimensions with the same target (e.g. holographic parity),
            # we need the impulse to reach ALL dimensions, not just the match between bit-index and dim-index.
            with torch.no_grad():
                nn.init.constant_(self.out_proj.weight, 1.0)
                nn.init.zeros_(self.out_proj.bias)
        else:
            # SIREN Network
            net = []
            net.append(SineLayer(self.coord_dim, hidden_dim, is_first=True, omega_0=self.omega_0))
            for _ in range(layers):
                net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=self.omega_0))
                
            self.net = nn.Sequential(*net)
            self.out_proj = nn.Linear(hidden_dim, emb_dim)
            
            # Proper SIREN Init (omega_0=30)
            with torch.no_grad():
                self.out_proj.weight.data *= 1.5 
                nn.init.zeros_(self.out_proj.bias)
            
        if self.mode == 'sinusoidal':
            # ensure even
            if coord_dim % 2 != 0: self.coord_dim += 1
            # Log-space frequencies for multi-scale resolution of the ID
            freqs = torch.exp(torch.arange(0, self.coord_dim, 2).float() * -(np.log(10000.0) / self.coord_dim))
            self.register_buffer('freqs', freqs)
        
    def forward(self, input_ids):
        """
        Forward pass through functional embedding.
        
        Args:
            input_ids: [batch, seq_len] - Token indices
            
        Returns:
            embeddings: [batch, seq_len, emb_dim] - Embedded vectors
        """
        B, L = input_ids.shape
        inputs = input_ids.unsqueeze(-1).float()
        
        if self.mode == 'binary' or self.mode == 'linear':
             # Convert IDs to Bits [B, L, coord_dim]
             mask = 2**torch.arange(self.coord_dim).to(input_ids.device)
             bits = (input_ids.unsqueeze(-1) & mask) > 0
             if self.mode == 'linear':
                 coords = bits.float() # Use {0, 1} for direct force channel
             else:
                 coords = bits.float() * 2 - 1 # Map {0, 1} to {-1, 1} for SIREN
        else:
            # Sinusoidal
            args = inputs * self.freqs
            coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # 2. Evaluate Field
        x_out = self.net(coords)
        out = self.out_proj(x_out)
        
        # 3. Apply Multiplier
        # Controlled by Level 12 impulse_scale interface
        out = out * self.impulse_scale
        
        # Enforce Zero-Input = Zero-Force (Critical for Inertial Memory tasks like Parity)
        # If all coordinate bits are 0 (ID=0), force should be 0.
        if self.mode == 'binary' or self.mode == 'linear':
         active_mask = (bits.float().sum(dim=-1, keepdim=True) > 0).float()
         out = out * active_mask
             
        return out
