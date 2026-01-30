"""
Toroidal Distance Loss
======================

3D toroidal geometry distance.
"""

import torch
import torch.nn as nn
from ..geometry.boundaries import toroidal_dist_python


def toroidal_distance_loss(x_pred, x_target):
    """
    Toroidal Distance Loss.
    
    Computes distance on 3D toroidal manifold.
    
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
