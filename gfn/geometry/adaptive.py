import torch
import torch.nn as nn
from ..constants import EPSILON_STANDARD, CURVATURE_CLAMP, FRICTION_SCALE

class AdaptiveRankChristoffel(nn.Module):
    """
    Adaptive Rank Christoffel Symbol Decomposition.
    Adjusts the effective rank of the curvature model dynamically based on 
    the input complexity (proxied by velocity norm).
    """
    def __init__(self, dim, max_rank=64, physics_config=None):
        super().__init__()
        self.dim = dim
        self.max_rank = max_rank
        self.config = physics_config or {}
        
        # Full-rank matrices (we'll slice them)
        self.U_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
        self.W_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
        
        # Rank predictor: Learns to estimate "geometric complexity"
        self.complexity_net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.return_friction_separately = False
        
    def forward(self, v, x=None, **kwargs):
        # 1. Predict rank ratio (0.1 to 1.0)
        # We use v.detach() to prevent complexity prediction from affecting v directly initially
        rank_ratio = 0.1 + 0.9 * self.complexity_net(v.detach())
        # We take the mean across the batch for rank slicing to keep tensor operations uniform
        # (Though we could do per-instance masks, slicing is more efficient if collective)
        avg_ratio = rank_ratio.mean().item()
        
        eff_rank = max(4, min(self.max_rank, int(avg_ratio * self.max_rank)))
        
        # 2. Slice parameters
        U = self.U_full[:, :eff_rank]
        W = self.W_full[:, :eff_rank]
        
        # 3. Standard Low-rank computation
        proj = torch.matmul(v, U)
        norm = torch.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm + EPSILON_STANDARD)
        sq = (proj * proj) * scale
        gamma = torch.matmul(sq, W.t())
        
        # Friction fallback (simplified for adaptive)
        mu = torch.zeros_like(v)
        
        if self.return_friction_separately:
            gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
            return gamma, mu
            
        return CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
