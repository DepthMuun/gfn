"""
Curiosity Loss
==============

Entropy-driven exploration loss for diverse geodesic learning.
"""

import torch


def curiosity_loss(velocities: list, lambda_c: float = 0.05) -> torch.Tensor:
    """
    Entropy-Driven Curiosity Loss (Thermodynamics).
    
    Encourages the model to explore diverse cognitive geodesics by maximizing 
    the differential entropy of the velocity distribution.
    
    Concept:
        Maximizing entropy prevents "cognitive collapse" and forces the model 
        to find new ways to resolve the same Hamiltonian task.
    
    Formula:
        S = Σ log(std(v_i) + ε)  (Entropy proxy for Gaussian-like latent distribution)
        L_C = - λ_c * S
        
    Args:
        velocities: List of velocity tensors [v_0, v_1, ..., v_T], each [batch, dim]
        lambda_c: Curiosity Temperature (T) - controls exploration strength
        
    Returns:
        Scalar loss tensor (negative entropy to minimize)
        
    Example:
        >>> v_seq = [torch.randn(32, 256) for _ in range(10)]
        >>> loss = curiosity_loss(v_seq, lambda_c=0.05)
        >>> # Lower loss = higher entropy = more exploration
    
    Reference:
        Inspired by thermodynamic entropy maximization principles
        for preventing mode collapse in latent representations.
    """
    if not velocities:
        return torch.tensor(0.0, device='cpu')
    
    if len(velocities) == 0:
        return torch.tensor(0.0, device=velocities[0].device if velocities else 'cpu')
        
    # Concatenate all velocities across time
    all_v = torch.cat(velocities, dim=0)  # [Batch * Seq, Dim]
    
    # Calculate batch-wise standard deviation for each dimension
    # We add epsilon for numerical stability of log
    v_std = all_v.std(dim=0) + 1e-6  # [Dim]
    
    # Entropy proxy: Sum of log-stds
    # Higher std = higher entropy = more exploration
    entropy = torch.log(v_std).sum()
    
    # We want to MAXIMIZE entropy, so we MINIMIZE negative entropy
    return -lambda_c * entropy
