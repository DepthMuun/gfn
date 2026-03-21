import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, List
from ...models.components.activations import SineLayer

class FunctionalEmbedding(nn.Module):
    """
    Core Logic for Functional/Implicit Embeddings.
    
    Modes:
      'lookup'     — Standard nn.Embedding (discrete tokens → vectors)
      'linear'     — Bit-expansion of token IDs → linear projection
      'binary'     — Bit-expansion normalized to [-1, 1]
      'siren'      — SIREN network for implicit coordinate encoding
      'continuous' — [P2.1] Direct continuous vector projection [B, T, D_in] → [B, T, D]
                     Enables native multimodal: images, audio, any continuous input.
                     Input does NOT go through vocabulary lookup — it's a direct force.
    """
    def __init__(self, vocab_size: int, emb_dim: int, coord_dim: int = 16, 
                 hidden_dim: int = 64, layers: int = 2, mode: str = 'linear', 
                 impulse_scale: float = 1.0, omega_0: float = 30.0,
                 continuous_input_dim: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.mode = mode
        self.coord_dim = coord_dim
        self.omega_0 = omega_0
        self.impulse_scale = nn.Parameter(torch.tensor(impulse_scale))
        
        if self.mode == 'continuous':
            # P2.1: Native multimodal — project continuous input [B, T, D_in] → [B, T, D]
            # D_in defaults to coord_dim if not specified (for backward compat)
            in_dim = continuous_input_dim if continuous_input_dim is not None else coord_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, emb_dim),
            )
            self.out_proj = nn.Identity()
        elif self.mode == 'lookup':
            self.net = nn.Embedding(vocab_size, emb_dim)
            self.out_proj = nn.Identity()
        elif self.mode == 'linear' or self.mode == 'binary':
            self.net = nn.Identity()
            self.out_proj = nn.Linear(self.coord_dim, emb_dim)
            nn.init.constant_(self.out_proj.weight, 1.0)
            nn.init.zeros_(self.out_proj.bias)
        else:
            # SIREN / MLP style
            net_layers: List[nn.Module] = []
            net_layers.append(SineLayer(self.coord_dim, hidden_dim, is_first=True, omega_0=self.omega_0))
            for _ in range(layers):
                net_layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=self.omega_0))
            self.net = nn.Sequential(*net_layers)
            self.out_proj = nn.Linear(hidden_dim, emb_dim)

        self.register_buffer('bit_mask', 2**torch.arange(self.coord_dim))

    def _map_coords(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == 'lookup':
            return inputs
            
        mask = self.bit_mask.to(inputs.device)
        bits = (inputs.long().bitwise_and(mask)) > 0
        coords = bits.float()
        
        if self.mode == 'binary':
            coords = coords * 2 - 1
        elif self.mode == 'sinusoidal':
            freqs = torch.exp(torch.arange(0, self.coord_dim, 2).to(inputs.device).float() * -(np.log(10000.0) / self.coord_dim))
            args = inputs.float() * freqs
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            
        return coords

    def forward(self, input_ids: torch.Tensor = None,
                continuous_input: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids:        Token indices [B, T] — for discrete modalities (text, XOR, etc.)
            continuous_input: Continuous vectors [B, T, D_in] — for images, audio, etc.
                             Only used when mode='continuous'.
        Returns:
            forces [B, T, D] — impulse forces in manifold space
        """
        if self.mode == 'continuous':
            if continuous_input is None:
                raise ValueError("FunctionalEmbedding(mode='continuous') requires continuous_input, not input_ids.")
            # Direct projection: [B, T, D_in] → [B, T, D]
            return self.net(continuous_input) * self.impulse_scale
            
        if self.mode == 'lookup':
            return self.net(input_ids) * self.impulse_scale
            
        inputs = input_ids.unsqueeze(-1)
        coords = self._map_coords(inputs)
        x = self.net(coords)
        out = self.out_proj(x)
        return out * self.impulse_scale

class StandardEmbedding(FunctionalEmbedding):
    """Standard Lookup Table Baseline."""
    def __init__(self, vocab_size: int, emb_dim: int, **kwargs):
        super().__init__(vocab_size, emb_dim, mode='lookup', **kwargs)

class BinaryEmbedding(FunctionalEmbedding):
    """Functional Embedding with Binary mapping [-1, 1]."""
    def __init__(self, vocab_size: int, emb_dim: int, **kwargs):
        super().__init__(vocab_size, emb_dim, mode='binary', **kwargs)
