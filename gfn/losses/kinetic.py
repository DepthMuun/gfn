"""
Kinetic Energy Penalty
======================

L2 regularization on velocities.
"""

import torch


def kinetic_energy_penalty(velocities: list, lambda_k: float = 0.001) -> torch.Tensor:
    """
    L2 Regularization on Velocities.
    
    Encourages the model to be 'lazy' and move only when necessary.
    
    Args:
        velocities: List of velocity tensors [batch, dim]
        lambda_k: Regularization coefficient
        
    Returns:
        Kinetic energy penalty scalar
    """
    if not velocities:
        return torch.tensor(0.0)
    v_norms = torch.stack([v.pow(2).sum(dim=-1).mean() for v in velocities])
    return lambda_k * v_norms.mean()
