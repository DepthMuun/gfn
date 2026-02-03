"""
SIREN Layer
===========

Sinusoidal Representation Networks for high-frequency detail.
"""

import torch
import torch.nn as nn
import numpy as np
from ..constants import SIREN_OMEGA_0


class SineLayer(nn.Module):
    """
    Linear Layer with Sinusoidal Activation (SIREN).
    
    High-frequency periodic activation allows fitting complex signals/embeddings.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        is_first: Whether this is the first layer (affects initialization)
        omega_0: Frequency parameter (default from constants)
    
    References:
        Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
        https://arxiv.org/abs/2006.09661
    
    Examples:
        >>> # First layer
        >>> layer1 = SineLayer(16, 64, is_first=True, omega_0=30.0)
        >>> 
        >>> # Hidden layer
        >>> layer2 = SineLayer(64, 64, is_first=False, omega_0=30.0)
    """
    
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=SIREN_OMEGA_0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights according to SIREN paper."""
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
        """
        Forward pass with sinusoidal activation.
        
        Args:
            x: Input tensor
            
        Returns:
            sin(omega_0 * W*x + b)
        """
        return torch.sin(self.omega_0 * self.linear(x))
