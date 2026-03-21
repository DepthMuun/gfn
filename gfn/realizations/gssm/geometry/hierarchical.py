import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple, Any
from ..geometry.base import BaseGeometry
from ..geometry.low_rank import LowRankRiemannianGeometry
from ..registry import register_geometry

@register_geometry('hierarchical')
class HierarchicalGeometry(BaseGeometry):
    """
    Multi-Scale Riemannian Geometry (Christoffel Mixture).
    Combines multiple geometries (typically Low-Rank) with different scales.
    
    Migrated from gfn_old HierarchicalRiemannianGeometry.
    """
    def __init__(self, dim: int, rank: int = 16, ranks: Optional[List[int]] = None, 
                 num_heads: int = 1, config: Optional[Any] = None, **kwargs):
        super().__init__(config)
        self.dim = dim
        self.ranks = ranks if ranks is not None else [8, 16, 32]
        if rank not in self.ranks:
            # Optionally include the factory-suggested rank
            self.ranks = sorted(list(set(self.ranks + [rank])))
        self.num_heads = num_heads
        
        # Initialize sub-geometries (defaulting to LowRank)
        self.scales = nn.ModuleList([
            LowRankRiemannianGeometry(dim, rank=r, num_heads=num_heads, config=config)
            for r in self.ranks
        ])
        
        # Learnable mixing weights
        self.scale_weights = nn.Parameter(torch.ones(len(self.ranks)) / len(self.ranks))
        self.return_friction_separately = False

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, 
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        gammas = []
        frictions = []
        
        # Execute each scale
        for scale in self.scales:
            # Temporarily ensure consistent return mode
            was_sep = getattr(scale, 'return_friction_separately', False)
            scale.return_friction_separately = True
            
            res = scale(x, v, force=force, **kwargs)
            if isinstance(res, tuple):
                g, f = res
            else:
                g, f = res, torch.zeros_like(v) if v is not None else torch.zeros_like(x)
            
            gammas.append(g)
            frictions.append(f)
            scale.return_friction_separately = was_sep

        # Mix using softmax weights
        weights = torch.softmax(self.scale_weights, dim=0)
        
        gamma_mixed = sum(w * g for w, g in zip(weights, gammas))
        friction_mixed = sum(w * f for w, f in zip(weights, frictions))
        
        if self.return_friction_separately:
            return gamma_mixed, friction_mixed
            
        if v is not None:
             return gamma_mixed + friction_mixed * v
        return gamma_mixed

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scale_weights, dim=0)
        metrics = [scale.metric_tensor(x) for scale in self.scales]
        return sum(w * m for w, m in zip(weights, metrics))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # Hierarchical usually just projects using the first scale (geometry consistent)
        from typing import cast
        return cast(BaseGeometry, self.scales[0]).project(x)

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scale_weights, dim=0)
        dists = [scale.dist(x1, x2) for scale in self.scales]
        return sum(w * d for w, d in zip(weights, dists))
