"""
Hamiltonian Energy Conservation Loss
=====================================

Riemannian Hamiltonian energy conservation for stable training.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. PHYSICS CONTRADICTION:
   External forces (from token embeddings) constantly inject energy into the system.
   Penalizing energy changes contradicts the physical response to forces.
   
2. SOLUTION - CONFIGURED MODES:
   - 'none': No energy conservation loss (recommended for standard training)
   - 'adaptive': Only penalize when external forces are small
   - 'structural': Penalize changes in energy STRUCTURE (smoother constraint)
   - 'relative': Penalize relative changes dE/E instead of absolute

3. RECOMMENDATION:
   For most tasks, use mode='none' or mode='adaptive'.
   The physics-grounded supervision should come from geodesic flow,
   not artificial energy conservation.
"""

import torch
from ..constants import EPSILON_SMOOTH


def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, 
                    lambda_h: float = 0.01, forces: list = None,
                    mode: str = 'adaptive') -> torch.Tensor:
    """
    Riemannian Hamiltonian Energy Conservation Loss.
    
    IMPORTANT (Auditoria 2026-02-06): This loss has CONFIGURED MODES due to
    the fundamental contradiction between energy conservation and external forces.
    
    Modes:
    - 'none': No energy conservation penalty (RECOMMENDED for standard training)
    - 'adaptive': Only penalize when external forces are small (|F| < threshold)
    - 'structural': Penalize changes in energy STRUCTURE (smoother, more stable)
    - 'relative': Penalize relative changes dE/E (scale-invariant)
    
    Args:
        velocities: List of velocity tensors [batch, dim]
        states: Optional list of position tensors [batch, dim]
        metric_fn: Optional metric function g(x)
        lambda_h: Loss coefficient
        forces: Optional list of force tensors for masking
        mode: Conservation mode ('none', 'adaptive', 'structural', 'relative')
        
    Returns:
        Energy conservation loss scalar
    """
    # AUDIT FIX: Handle 'none' mode efficiently
    if mode == 'none' or lambda_h == 0.0 or not velocities or len(velocities) < 2:
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
    
    if mode == 'adaptive':
        # AUDIT FIX: Adaptive mode - only penalize when forces are small
        # This respects the physics: forces SHOULD change energy
        for i in range(len(energies) - 1):
            dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
            if forces is not None and i < len(forces):
                f_norm = forces[i].pow(2).sum(dim=-1)
                # Only penalize energy change when external force is negligible
                force_threshold = 1e-4
                mask = (f_norm < force_threshold).float()
                dE = dE * mask
            diffs.append(dE)
            
    elif mode == 'structural':
        # AUDIT FIX: Structural mode - penalize changes in energy distribution
        # This is a softer constraint that doesn't fight against external forces
        for i in range(len(energies) - 1):
            # Compare energy distributions rather than absolute changes
            # This captures whether the SHAPE of energy changes is reasonable
            e_curr = energies[i]
            e_next = energies[i + 1]
            
            # Use smooth L1 for structural similarity
            diff = torch.abs(e_curr - e_next) / (torch.abs(e_curr) + EPSILON_SMOOTH)
            diffs.append(diff)
            
    elif mode == 'relative':
        # AUDIT FIX: Relative mode - penalize fractional energy change
        # This is scale-invariant and more interpretable
        for i in range(len(energies) - 1):
            e_curr = energies[i]
            e_next = energies[i + 1]
            
            # Relative change with epsilon for numerical stability
            denom = torch.abs(e_curr) + EPSILON_SMOOTH
            rel_change = torch.abs(e_next - e_curr) / denom
            
            # Use smooth approximation
            diff = torch.sqrt(rel_change.pow(2) + EPSILON_SMOOTH)
            diffs.append(diff)
            
    else:
        # Fallback to original behavior
        for i in range(len(energies) - 1):
            dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
            diffs.append(dE)
        
    return lambda_h * torch.stack(diffs).mean()
