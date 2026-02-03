"""
Manifold SGD Optimizer
======================

Simple SGD with retraction for Riemannian manifolds.
"""

import torch
from torch.optim import Optimizer


class ManifoldSGD(Optimizer):
    """
    Simple SGD with retraction for Riemannian manifolds.
    
    Uses the same retraction concept but without momentum.
    Useful for debugging or when Adam is unstable.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-2)
        weight_decay: L2 regularization (default: 0.0)
        max_norm: Maximum weight norm for retraction (default: 10.0)
    
    Examples:
        >>> # Basic usage
        >>> optimizer = ManifoldSGD(model.parameters(), lr=1e-2)
        
        >>> # With weight decay
        >>> optimizer = ManifoldSGD(
        ...     model.parameters(),
        ...     lr=1e-2,
        ...     weight_decay=0.01,
        ...     max_norm=5.0
        ... )
    """
    
    def __init__(self, params, lr=1e-2, weight_decay=0.0, max_norm=10.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, max_norm=max_norm)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: Optional closure to reevaluate the model
            
        Returns:
            loss: Optional loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            max_norm = group['max_norm']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update with retraction
                p.data.add_(grad, alpha=-lr)
                
                # Normalize if needed
                norm = p.data.norm()
                if norm > max_norm:
                    p.data.mul_(max_norm / norm)
        
        return loss
