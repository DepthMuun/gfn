"""
Geodesic Curvature Regularization
==================================

Regularization based on Christoffel symbols.
"""

import torch
from ..constants import GEODESIC_FUSED_SCALE


def geodesic_regularization(velocities: list, christoffel_outputs: list, 
                           lambda_g: float = 0.001) -> torch.Tensor:
    """
    Geodesic Curvature Regularization.
    
    Supports both standard list of tensors and fused pre-computed sum.
    
    Args:
        velocities: List of velocity tensors (unused, kept for API compatibility)
        christoffel_outputs: List of Christoffel symbol tensors
        lambda_g: Regularization coefficient
        
    Returns:
        Curvature regularization loss scalar
    """
    if not christoffel_outputs:
        return torch.tensor(0.0)
    
    # Check if this is a fused regulation tensor (single tensor in list)
    if len(christoffel_outputs) == 1 and christoffel_outputs[0].dim() == 1:
        # Fused case: christoffel_outputs[0] is sum(||Gamma||^2) per batch item
        # To match all_curvatures.pow(2).mean(), we must divide by total elements per batch item
        # This isn't strictly known here, but we can pass it or retrieve it.
        # For MANIFOLD models, this is normally (depth * seq_len * dim)
        # We'll use a conservative estimate or let it be scaled by lambda_g.
        # Actually, let's keep it simple as a sum for now, but scaled.
        return lambda_g * christoffel_outputs[0].mean() / GEODESIC_FUSED_SCALE  # Heuristic scaling
    
    # Standard Vectorization:
    all_curvatures = torch.stack(christoffel_outputs) # [N_heads*Seq, B, d]
    curvature_norms = all_curvatures.pow(2).mean()
    return lambda_g * curvature_norms
