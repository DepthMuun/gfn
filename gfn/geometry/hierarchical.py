import torch
import torch.nn as nn
from .lowrank import LowRankChristoffel

class HierarchicalChristoffel(nn.Module):
    """
    Multi-Scale Christoffel Symbolic Mixture.
    Combines multiple LowRankChristoffel modules with different ranks 
    to capture both local (digit-level) and global (hierarchy-level) 
    geometric features.
    """
    def __init__(self, dim, ranks=[8, 16, 32], physics_config=None):
        super().__init__()
        self.dim = dim
        self.ranks = ranks
        self.scales = nn.ModuleList([
            LowRankChristoffel(dim, rank=r, physics_config=physics_config)
            for r in ranks
        ])
        
        # Learnable mixing weights
        self.scale_weights = nn.Parameter(torch.ones(len(ranks)) / len(ranks))
        
        # We also need a shared friction/singularity gate (optional, let's use the highest rank's)
        self.return_friction_separately = False # Will be set by MLayer
        
    def forward(self, v, x=None, force=None, **kwargs):
        gammas = []
        mus = []
        
        # Compute all scales
        for scale in self.scales:
            # Temporarily ensure they all return parts separately
            old_val = getattr(scale, 'return_friction_separately', False)
            scale.return_friction_separately = True
            
            gamma, mu = scale(v, x, force, **kwargs)
            gammas.append(gamma)
            mus.append(mu)
            
            scale.return_friction_separately = old_val
            
        # Softmax mixing
        weights = torch.softmax(self.scale_weights, dim=0)
        
        gamma_combined = sum(w * g for w, g in zip(weights, gammas))
        # Use average friction or highest-rank friction
        mu_combined = sum(w * m for w, m in zip(weights, mus))
        
        if self.return_friction_separately:
            return gamma_combined, mu_combined
            
        return gamma_combined + mu_combined * v
