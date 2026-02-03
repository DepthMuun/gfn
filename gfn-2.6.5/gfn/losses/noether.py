"""
Noether Symmetry Loss
=====================

Semantic symmetry loss for isomeric head groups.
"""

import torch


def noether_loss(christoffel_outputs: list, isomeric_groups: list = None, 
                lambda_n: float = 0.01) -> torch.Tensor:
    """
    Semantic Symmetry (Noether) Loss.
    
    Enforces that 'Isomeric' subspaces (heads) learn the same geometric laws
    even if their specific weights are not strictly tied (Soft Symmetry).
    
    If weights ARE hard-tied (Isomeric Heads in MLayer), this term acts as a 
    regularizer to ensure gradients are consistent across symmetric contexts.
    
    Args:
        christoffel_outputs: List of Γ(v) outputs per head
        isomeric_groups: List of head index groups [[0, 1], [2, 3]]
        lambda_n: Noether coefficient
        
    Returns:
        Symmetry loss scalar
    """
    if not isomeric_groups or not christoffel_outputs:
        return torch.tensor(0.0, device=christoffel_outputs[0].device if christoffel_outputs else 'cpu')
        
    total_diff = 0.0
    count = 0
    
    for group in isomeric_groups:
        if len(group) < 2: continue
        
        # Reference head output in this group
        ref_out = christoffel_outputs[group[0]]
        
        for other_h_idx in group[1:]:
            target_out = christoffel_outputs[other_h_idx]
            # MSE between geometric responses of symmetric heads
            total_diff = total_diff + torch.mean((ref_out - target_out).pow(2))
            count += 1
            
    if count == 0:
        return torch.tensor(0.0, device=christoffel_outputs[0].device)
        
    return lambda_n * (total_diff / count)
