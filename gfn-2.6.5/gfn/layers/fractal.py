import torch
import torch.nn as nn
from .base import MLayer

class FractalMLayer(nn.Module):
    """
    Fractal Manifold Layer: Implements multiscale "Recursive Tunneling".
    
    If local curvature R is high, the particle "tunnels" into a 
    high-resolution sub-manifold to resolve semantic complexity.
    """
    def __init__(self, dim, heads=8, rank=16, base_dt=0.1, integrator_type='symplectic', physics_config=None, layer_idx=0, total_depth=6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.rank = rank
        self.physics_config = physics_config or {}
        
        # DeepNet-style depth scaling for gradient stability
        self.layer_idx = layer_idx
        self.total_depth = total_depth
        self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth
        
        # Macro-manifold: Standard MLayer evolution
        self.macro_manifold = MLayer(
            dim, heads=heads, rank=rank, 
            base_dt=base_dt, integrator_type=integrator_type, 
            physics_config=self.physics_config
        )
        
        # Sub-manifold: Dedicated to resolving high-curvature details
        # Smaller rank but higher resolution (smaller dt)
        micro_cfg = self.physics_config.copy()
        # Disable fractal recursion in the sub-manifold to avoid infinite loops
        if 'fractal' not in micro_cfg: micro_cfg['fractal'] = {}
        micro_cfg['fractal']['enabled'] = False 
        
        self.micro_manifold = MLayer(
            dim, heads=heads, rank=max(8, rank//2), 
            base_dt=base_dt * 0.5, integrator_type=integrator_type, 
            physics_config=micro_cfg
        )
        
        fract_cfg = self.physics_config.get('fractal', {})
        self.threshold = fract_cfg.get('threshold', 0.5)
        self.alpha_scale = fract_cfg.get('alpha', 0.2)
        
    def forward(self, x, v, force=None, context=None, collect_christ=False):
        # 1. Macro-evolution (Standard flow)
        x_m, v_m, ctx_m, christoffels = self.macro_manifold(x, v, force, context, collect_christ=collect_christ)
        
        if not self.physics_config.get('fractal', {}).get('enabled', False):
            return x_m, v_m, ctx_m, christoffels
            
        # 2. Estimate average Curvature R from Christoffel magnitudes
        # Gamma has shape [batch, head_dim]
        # We stack and take the norm to estimate local complexity
        stacked_gamma = torch.stack(christoffels, dim=1) # [batch, heads, head_dim]
        curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]
        
        # tunnel_gate = 0.3 + 0.7 * torch.sigmoid((curvature_r - self.threshold) * 5.0)
        # LEVEL 42: SURGICAL NOISE REMOVAL
        # Removing the 0.3 bias ensures that flat regions (R=0) don't inject micro-manifold noise.
        tunnel_gate = torch.sigmoid((curvature_r - self.threshold) * 10.0) 
        
        # 4. Micro-evolution (Zooming in)
        # We use the macro-updated state as input to the sub-manifold
        # to refine the results in complex semantic regions.
        x_f, v_f, _, _ = self.micro_manifold(x_m, v_m, force, context, collect_christ=collect_christ)
        
        # 5. Recursive Blending
        # The micro-manifold provides a perturbative correction to the macro-flow
        x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale
        v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale
        
        return x_final, v_final, ctx_m, christoffels
