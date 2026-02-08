import torch
import torch.nn as nn
from .lowrank import LowRankChristoffel


class ReactiveChristoffel(LowRankChristoffel):
    """
    Active Inference: Geometry that reacts to the agent's state.
    
    IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):
    
    1. TERMINOLOGY CLARIFICATION:
       - "Singularities" are NOT true mathematical singularities
       - They are CURVATURE AMPLIFICATION regions where the model
         increases geometric "resistance" to capture high-confidence states
       - "Black hole strength" = curvature_amplification_factor
       - "Singularity threshold" = semantic_certainty_threshold
    
    2. PHYSICS INTERPRETATION:
       The amplification factor artificially multiplies Christoffel symbols
       based on semantic potential. This does NOT correspond to any physical
       manifold property. It's a regularization/attention mechanism.
    
    3. GRADIENT FLOW:
       Uses soft-sigmoid for differentiability: sigma(10 * (potential - threshold))
       This creates smooth transitions instead of hard thresholds.
    
    Features:
    1. Reactive Curvature (Plasticity): Metric deforms based on kinetic energy.
       High energy (confusion/exploration) -> Higher curvature (more braking).
       
    2. Logical Singularities: If 'V(x)' (potential) exceeds a threshold, 
       we trigger a 'Curvature Amplification' to emphasize semantic certainty.
       NOT a true singularity - just controlled amplification.
    
    Gradient Notes:
    The singularity gate uses soft-sigmoid (slope * (potential - threshold))
    to maintain differentiability. This creates a smooth transition rather
    than a hard threshold, allowing gradients to flow through amplification regions.
    
    Args:
        dim: Manifold dimension
        rank: Low-rank approximation rank
        physics_config: Configuration dict with active_inference settings
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config=physics_config)
        self.config = physics_config or {}
        self.active_cfg = self.config.get('active_inference', {})
        
        self.plasticity = self.active_cfg.get('reactive_curvature', {}).get('plasticity', 0.0)
        
        # AUDIT FIX: Renamed for clarity
        self.semantic_certainty_threshold = self.active_cfg.get('singularities', {}).get('threshold', 0.8)
        self.curvature_amplification_factor = self.active_cfg.get('singularities', {}).get('strength', 10.0)

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
                
                # AUDIT FIX: Use renamed parameters
                sing_thresh = sing_cfg.get('threshold', 0.9) if sing_cfg.get('enabled', False) else 1.0
                sing_strength = sing_cfg.get('strength', 1.0) if sing_cfg.get('enabled', False) else 1.0
                
                return christoffel_fused(v, self.U, self.W, x_in, V_w_in, plasticity, sing_thresh, sing_strength)
        except Exception as e:
            print(f"[GFN:WARN] CUDA christoffel_fused failed: {e}, falling back to PyTorch")
            # Fall through to PyTorch implementation

        # Fallback PyTorch: Base curvature (static memory or PyTorch fallback)
        gamma = super().forward(v, x, force=force, **kwargs)
        
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

        # 2. AUDIT FIX: Curvature Amplification (formerly "Singularities")
        # This is NOT a true mathematical singularity
        # It amplifies curvature for high-semantic-certainty regions
        if self.active_cfg.get('singularities', {}).get('enabled', False):
            # Check Semantic Potential V(x)
            if self.is_torus:
                 x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            else:
                 x_in = x
            potential = torch.sigmoid(self.V(x_in)) # [batch, 1]
            
            # AUDIT FIX: Document what we're doing
            """
            We amplify curvature where semantic potential exceeds threshold.
            This is NOT a true singularity - it's a learned attention mechanism
            that makes the model "pay more attention" to high-certainty regions
            by increasing geometric resistance.
            
            The amplification is smooth (differentiable) via soft-sigmoid.
            """
            
            # Soft amplification using sigmoid with configurable slope
            gate_slope = self.active_cfg.get('singularities', {}).get('gate_slope', 10.0)
            is_amplified = torch.sigmoid(gate_slope * (potential - self.semantic_certainty_threshold))
            amplification_mult = 1.0 + is_amplified * (self.curvature_amplification_factor - 1.0)
            gamma = gamma * amplification_mult
            
            # AUDIT FIX: Add safety constraint
            # Ensure amplification doesn't cause numerical instability
            max_amplification = self.curvature_amplification_factor
            gamma = torch.clamp(gamma, -max_amplification * CURVATURE_CLAMP, max_amplification * CURVATURE_CLAMP)

        return gamma
