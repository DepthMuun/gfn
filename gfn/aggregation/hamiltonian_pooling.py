"""
Hamiltonian Pooling
===================

Energy-weighted aggregation of sequence states.

Tokens with higher Hamiltonian energy (kinetic + potential) 
contribute more to the final aggregated state.

Physics Motivation:
- High-energy states are more "important" in dynamical systems
- Energy indicates strength of interaction/force application
- Natural weighting scheme from first principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HamiltonianPooling(nn.Module):
    """
    Energy-weighted aggregation using Hamiltonian dynamics.
    
    H = K + U where:
    - K = (1/2) v^T g v  (kinetic energy)
    - U = (1/2) ||x||^2  (potential energy)
    
    States with higher H get higher attention weight.
    
    Args:
        dim: State dimension
        temperature: Softmax temperature for energy weighting (default: 1.0)
        learn_metric: If True, learn Riemannian metric for kinetic energy (default: False)
    
    Example:
        >>> pooling = HamiltonianPooling(dim=128, temperature=0.5)
        >>> x_agg, v_agg, weights = pooling(x_seq, v_seq)
        >>> # weights[i] is higher for high-energy tokens
    """
    
    def __init__(self, dim, temperature=1.0, learn_metric=False):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.learn_metric = learn_metric
        
        if learn_metric:
            # Learnable diagonal metric g_ii
            self.metric = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('metric', torch.ones(dim))
    
    def kinetic_energy(self, v):
        """
        Compute kinetic energy: K = 0.5 * v^T @ g @ v
        
        Args:
            v: Velocity [B, L, dim]
        
        Returns:
            K: Kinetic energy [B, L]
        """
        # Ensure metric is a tensor and broadcast to match v shape
        B, L, dim = v.shape
        metric_expanded = self.metric.view(1, 1, -1).expand(B, L, -1)  # [B, L, dim]
        weighted_v = v * metric_expanded  # [B, L, dim]
        K = 0.5 * (v * weighted_v).sum(dim=-1)  # [B, L]
        return K
    
    def potential_energy(self, x):
        """
        Compute potential energy: U = 0.5 * ||x||^2
        
        Simple quadratic potential. Can be extended to learned potential.
        
        Args:
            x: Position [B, L, dim]
        
        Returns:
            U: Potential energy [B, L]
        """
        U = 0.5 * (x ** 2).sum(dim=-1)  # [B, L]
        return U
    
    def forward(self, x_seq, v_seq):
        """
        Aggregate states weighted by Hamiltonian energy.
        
        Args:
            x_seq: Position sequence [B, L, dim]
            v_seq: Velocity sequence [B, L, dim]
        
        Returns:
            x_agg: Aggregated position [B, dim]
            v_agg: Aggregated velocity [B, dim]
            weights: Attention weights [B, L] for interpretability
        """
        B, L, dim = x_seq.shape
        
        # Compute total energy per token
        K = self.kinetic_energy(v_seq)  # [B, L]
        U = self.potential_energy(x_seq)  # [B, L]
        H = K + U  # [B, L] - Total Hamiltonian
        
        # Softmax weighting (high energy → high weight)
        weights = F.softmax(H / self.temperature, dim=-1)  # [B, L]
        
        # Weighted aggregation
        x_agg = (weights.unsqueeze(-1) * x_seq).sum(dim=1)  # [B, dim]
        v_agg = (weights.unsqueeze(-1) * v_seq).sum(dim=1)  # [B, dim]
        
        return x_agg, v_agg, weights
