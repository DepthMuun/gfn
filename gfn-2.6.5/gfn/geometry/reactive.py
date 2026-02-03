import torch
import torch.nn as nn
from .lowrank import LowRankChristoffel

class ReactiveChristoffel(LowRankChristoffel):
    """
    Active Inference: Geometry that reacts to the agent's state.
    
    Features:
    1. Reactive Curvature (Plasticity): Metric deforms based on kinetic energy.
       High energy (confusion/exploration) -> Higher curvature (more braking).
       
    2. Logical Singularities: If 'V(x)' (potential) exceeds a threshold, 
       we trigger a 'Black Hole' (infinite curvature) to trap the thought 
       in a semantic certainty.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config=physics_config)
        self.config = physics_config or {}
        self.active_cfg = self.config.get('active_inference', {})
        
        self.plasticity = self.active_cfg.get('reactive_curvature', {}).get('plasticity', 0.0)
        self.singularity_threshold = self.active_cfg.get('singularities', {}).get('threshold', 0.8)
        self.black_hole_strength = self.active_cfg.get('singularities', {}).get('strength', 10.0)

    def forward(self, v, x=None, force=None, **kwargs):
        # Try CUDA path with Active Inference
        try:
            from gfn.cuda.ops import christoffel_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and v.is_cuda:
                # Extract Active Inference parameters
                x_in = x if x is not None else torch.empty(0, device=v.device)
                
                # Singularities require V_w  
                sing_cfg = self.active_cfg.get('singularities', {})
                if sing_cfg.get('enabled', False) and x is not None:
                    V_w_in = self.V.weight.t()  # [1, dim] -> [dim, 1] -> [1, dim]
                else:
                    V_w_in = torch.empty(0, device=v.device)
                
                # Plasticity
                react_cfg = self.active_cfg.get('reactive_curvature', {})
                plasticity = self.plasticity if react_cfg.get('enabled', False) else 0.0
                
                # Singularity params
                sing_thresh = sing_cfg.get('threshold', 0.9) if sing_cfg.get('enabled', False) else 1.0
                sing_strength = sing_cfg.get('strength', 1.0) if sing_cfg.get('enabled', False) else 1.0
                
                return christoffel_fused(v, self.U, self.W, x_in, V_w_in, plasticity, sing_thresh, sing_strength)
        except Exception as e:
            print(f"[GFN:WARN] CUDA christoffel_fused failed: {e}, falling back to PyTorch")
            # Fall through to PyTorch implementation

        # Fallback PyTorch: Base curvature (static memory or PyTorch fallback)
        gamma = super().forward(v, x, force=force)
        
        if not self.active_cfg.get('enabled', False):
            return gamma
            
        # 1. Reactive Curvature (Plasticity)
        if self.active_cfg.get('reactive_curvature', {}).get('enabled', False):
            # Energy = Kinetic Energy of thoughts (~ v^2)
            # Use tanh to bound the reaction
            energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
            # If energy is high, increase curvature (slow down/turn harder)
            # Gamma_new = Gamma * (1 + alpha * energy)
            gamma = gamma * (1.0 + self.plasticity * energy)
            
        # 2. Logical Singularities (Black Holes)
        if self.active_cfg.get('singularities', {}).get('enabled', False):
            # Check Semantic Potential V(x)
            if self.is_torus:
                 x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            else:
                 x_in = x
            potential = torch.sigmoid(self.V(x_in)) # [batch, 1]
            
            # If we are very sure (High Potential), trigger Singularity
            # GRADIENT FIX: Use soft-sigmoid instead of hard threshold for differentiability
            is_singularity = torch.sigmoid(10.0 * (potential - self.singularity_threshold))
            singularity_mult = 1.0 + is_singularity * (self.black_hole_strength - 1.0)
            gamma = gamma * singularity_mult
            
        return gamma
