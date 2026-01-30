"""
Hamiltonian Energy Conservation Loss
=====================================

Riemannian Hamiltonian energy conservation for stable training.
"""

import torch
from ..constants import EPSILON_SMOOTH


def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, 
                    lambda_h: float = 0.01, forces: list = None) -> torch.Tensor:
    """
    Riemannian Hamiltonian Energy Conservation Loss.
    
    If 'metric_fn' is provided, computes Energy = 0.5 * v^T g(x) v.
    Otherwise fallbacks to Euclidean Energy = 0.5 * ||v||^2.
    
    Args:
        velocities: List of velocity tensors [batch, dim]
        states: Optional list of position tensors [batch, dim]
        metric_fn: Optional metric function g(x)
        lambda_h: Loss coefficient
        forces: Optional list of force tensors for masking
        
    Returns:
        Energy conservation loss scalar
    """
    if lambda_h == 0.0 or not velocities or len(velocities) < 2:
        return torch.tensor(0.0, device=velocities[0].device if (velocities and len(velocities) > 0) else 'cpu')
    
    energies = []
    for i in range(len(velocities)):
        v = velocities[i]
        if metric_fn is not None and states is not None:
             x = states[i]
             # E = 0.5 * sum(g_ii * v_i^2) for diagonal metrics
             g = metric_fn(x) 
             e = 0.5 * torch.sum(g * v.pow(2), dim=-1)
        else:
             e = 0.5 * v.pow(2).sum(dim=-1)
        energies.append(e)
        
    diffs = []
    for i in range(len(energies) - 1):
        # Use smooth L2 approximation instead of abs() to prevent gradient vanishing
        # sqrt(x^2 + eps) has non-zero gradient everywhere, unlike abs(x)
        dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
        if forces is not None and i < len(forces):
            f_norm = forces[i].pow(2).sum(dim=-1)
            mask = (f_norm < 1e-4).float()
            dE = dE * mask
        diffs.append(dE)
        
    return lambda_h * torch.stack(diffs).mean()
