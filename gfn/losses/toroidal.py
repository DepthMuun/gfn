"""
Toroidal Distance Loss
======================

Computes geodesic distance on a 3D toroidal manifold.

The toroidal topology wraps both angular coordinates to [0, 2π),
so the shortest distance between two points accounts for wrapping.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. FLAT TORUS vs LEARNED TORUS:
   This loss computes distance on a FLAT torus (product of circles):
   diff = min(|x1 - x2|, 2π - |x1 - x2|)
   dist = sqrt(sum(diff^2))
   
   This does NOT account for the Christoffel curvature learned by the model.
   For true geodesic distance on the learned manifold, use Christoffel-based
   distance computation.

2. USE CASES:
   - Good for: Regularization, comparing latent representations
   - Not suitable for: True geodesic loss on curved manifold

3. RECOMMENDATION:
   Use this loss for auxiliary tasks (regularization, clustering) but not
   as the primary supervision signal for manifold structure.

Mathematical formulation:
    For two points x1, x2 on torus:
    diff = min(|x1 - x2|, 2π - |x1 - x2|)
    dist = sqrt(sum(diff^2))

Example:
    >>> loss_fn = ToroidalDistanceLoss()
    >>> x_pred = torch.tensor([[0.1, 3.1], [2.0, 1.0]])  # On torus [0, 2π)
    >>> x_target = torch.tensor([[0.2, 3.0], [2.1, 1.1]])
    >>> loss = loss_fn(x_pred, x_target)
"""

import torch
import torch.nn as nn
from ..geometry.boundaries import toroidal_dist_python


def toroidal_distance_loss(x_pred, x_target):
    """
    Toroidal Distance Loss.
    
    Computes distance on 3D toroidal manifold (FLAT torus).
    
    AUDIT NOTE: This is distance on a flat torus, not the learned manifold.
    
    Args:
        x_pred: Predicted positions [batch, dim]
        x_target: Target positions [batch, dim]
        
    Returns:
        Toroidal distance loss scalar
    """
    dist = toroidal_dist_python(x_pred, x_target)
    return dist.pow(2).mean()


class ToroidalDistanceLoss(nn.Module):
    """nn.Module wrapper for toroidal_distance_loss."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x_target):
        return toroidal_distance_loss(x_pred, x_target)
