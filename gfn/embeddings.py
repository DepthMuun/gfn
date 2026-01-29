"""
Implicit Neural Embeddings (INFs)
=================================

Replaces discrete lookup tables with continuous functions defined on a manifold.
Based on SIREN (Sinusoidal Representation Networks) for high-frequency detail.

Theory:
Instead of storing a vector E[i] for every token i, we store a low-rank coordinate c[i]
and learn a continuous function f(c) -> R^D.

    Embedding(i) = f( Coord(i) )

This allows:
1. Infinite Vocabulary (via hashing or continuous inputs)
2. Smooth Interpolation (Metric Topology between tokens)
3. Massive Parameter Reduction (O(1) vs O(V))
"""

import torch
import torch.nn as nn
import numpy as np
from .constants import EMBEDDING_SCALE, SIREN_OMEGA_0

class SineLayer(nn.Module):
    """
    Linear Layer with Sinusoidal Activation (SIREN).
    High-frequency periodic activation allows fitting complex signals/embeddings.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=SIREN_OMEGA_0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer needs range to cover multiple periods [-1, 1] -> [-omega, omega]
                bound = 1 / self.linear.weight.size(1)
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Subsequent layers need specific initialization for gradient flow consistency
                bound = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitEmbedding(nn.Module):
    """
    Implicit Neural Field Embedding.
    
    Maps Token IDs -> Learnable Coordinates -> Vector Space via SIREN.
    
    Args:
        vocab_size (int): Number of tokens (for coordinate table size).
        emb_dim (int): Output embedding dimension.
        coord_dim (int): Dimension of the underlying coordinate space (default: 16).
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
        Args:
            input_ids: [batch, seq_len] (Indices)
        Returns:
            embeddings: [batch, seq_len, emb_dim]
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

class FunctionalEmbedding(nn.Module):
    """
    Pure Functional Embedding (Zero-Lookup).
    Maps Index -> Coordinate -> SIREN -> Vector.
    
    Modes:
    - 'sinusoidal': High-freq hash (Good for input uniqueness, bad for readout).
    - 'binary': Bitwise representation (Good for learning/readout).
    - 'linear': Direct coordinate mapping (Essential for Parity/Arithmetic).
    
    O(1) Memory: Parameters do NOT scale with Vocab Size.
    """
    def __init__(self, vocab_size, emb_dim, coord_dim=16, hidden_dim=64, layers=2, mode='binary', impulse_scale=1.0, omega_0=SIREN_OMEGA_0):
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
        Args:
            input_ids: [batch, seq_len]
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
