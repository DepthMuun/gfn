"""
Circular Distance Loss
======================

Holographic phase loss on flat torus.
"""

import torch
import torch.nn as nn


def circular_distance_loss(x_pred, x_target):
    """
    Holographic Phase Loss (Level 13).
    
    Computes distance on the flat Torus T^n:
    L = 1 - cos(x_pred - x_target)
    
    Properties:
    1. Bounded [0, 2]
    2. Continuous at 2pi boundary
    3. Gradient is sin(delta), naturally clipped [-1, 1]
    
    Args:
        x_pred: Predicted positions [batch, dim]
        x_target: Target positions [batch, dim]
        
    Returns:
        Circular distance loss scalar
    """
    delta = x_pred - x_target
    return (1.0 - torch.cos(delta)).mean()


class CircularDistanceLoss(nn.Module):
    """nn.Module wrapper for circular_distance_loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x_pred, x_target):
        return circular_distance_loss(x_pred, x_target)
