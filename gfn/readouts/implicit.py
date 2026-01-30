"""
Implicit Readout Strategy
=========================

Temperature-annealed sigmoid readout with Gumbel-Softmax variant.
Supports both flat and toroidal topologies.
"""

import torch
import torch.nn as nn
from ..constants import READOUT_GAIN


class ImplicitReadout(nn.Module):
    """
    Temperature-Annealed Sigmoid Readout (Gumbel-Softmax variant)
    
    Prevents gradient cliffs from hard thresholding by using soft sigmoid
    with temperature that anneals from high (smooth) to low (sharp).
    
    Args:
        dim: Input dimension
        coord_dim: Output coordinate dimension  
        temp_init: Initial temperature (high = smooth gradients)
        temp_final: Final temperature (low = sharp outputs)
        topology: 0 for flat, 1 for toroidal
    
    Examples:
        >>> # Flat topology
        >>> readout = ImplicitReadout(dim=256, coord_dim=128, topology=0)
        >>> logits = readout(x)  # [batch, seq, 128]
        
        >>> # Toroidal topology
        >>> readout = ImplicitReadout(dim=256, coord_dim=128, topology=1)
        >>> logits = readout(x)  # Uses [sin(x), cos(x)] embedding
    """
    
    def __init__(self, dim, coord_dim, temp_init=1.0, temp_final=0.2, topology=0):
        super().__init__()
        
        self.topology = topology
        self.is_torus = (topology == 1)
        
        # MLP to output coordinates
        # Toroidal input uses [sin(x), cos(x)]
        in_dim = dim * 2 if self.is_torus else dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim), 
            nn.GELU(),
            nn.Linear(dim, coord_dim)
        )
        
        # Mark readout layers for init scaling in model.py
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.is_readout = True
        
        # Temperature annealing parameters
        self.temp_init = temp_init
        self.temp_final = temp_final
        
        # Training progress tracker
        self.register_buffer('training_step', torch.tensor(0))
        self.register_buffer('max_steps', torch.tensor(1000))
        
    def forward(self, x):
        """
        Forward pass with topology-aware embedding.
        
        Args:
            x: [batch, seq, dim] - Hidden states
            
        Returns:
            logits: [batch, seq, coord_dim] - Output logits
        """
        if self.is_torus:
            # Map x -> [sin(x), cos(x)] to enforce periodicity
            x_emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            x_emb = x
        
        # Gain provides sharper gradients for BCE
        logits = self.mlp(x_emb) * READOUT_GAIN  # Sharpen logits for better BCE loss
        
        return logits
        # Sigmoid is applied by the caller when needed
    
    def update_step(self):
        """Call this after each optimizer step to update temperature schedule."""
        if self.training:
            self.training_step += 1
    
    def set_max_steps(self, max_steps):
        """Update max steps for temperature schedule."""
        self.max_steps = torch.tensor(max_steps)
