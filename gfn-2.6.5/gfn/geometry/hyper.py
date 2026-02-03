import torch
import torch.nn as nn
from .lowrank import LowRankChristoffel

class HyperChristoffel(LowRankChristoffel):
    """
    Hyper-Christoffel: Context-Dependent Geometry.
    
    Architecture:
    Gamma(v, v | x) = W(x) * (U(x)^T v)^2
    
    Efficient Implementation (Gated Modulation):
    U(x) = U_static * diag(Gate_u(x))
    W(x) = W_static * diag(Gate_w(x))
    
    Where Gate(x) outputs a [rank] vector in [0, 2], scaling the importance 
    of each geometric basis vector dynamically.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config)
        
        # HyperNetworks: State x -> Modulation Gates [rank]
        # Light-weight: just a linear projection + activation
        self.gate_u = nn.Linear(dim, rank)
        self.gate_w = nn.Linear(dim, rank)
        
        # Initialize gates to be near identity (output ~1.0)
        # Sigmoid(0) = 0.5 -> * 2 = 1.0
        nn.init.zeros_(self.gate_u.weight)
        nn.init.zeros_(self.gate_u.bias)
        nn.init.zeros_(self.gate_w.weight)
        nn.init.zeros_(self.gate_w.bias)
        
    def forward(self, v, x=None):
        if x is None:
            # Fallback to static if no context provided (e.g. init or blind mode)
            return super().forward(v, None)
            
        # 1. Compute Context Gates
        # Range: [0, 2] - allowing to silence (0) or amplify (2) specific basis vectors
        g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]
        g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]
        
        # 2. Modulate Static Basis
        # U: [dim, rank]
        # g_u: [batch, rank]
        # Effective U: U * g_u (broadcast) -> effectively specific U for each batch item!
        # U_dynamic = U (1, dim, rank) * g_u (batch, 1, rank)
        
        # PyTorch optimization: Don't materialize full U_dynamic [batch, dim, rank] (too big)
        # Instead, modulate projection:
        # proj = v @ U -> [batch, rank]
        # proj_dynamic = proj * g_u
        
        # Weights U, W are [dim, rank]
        # v: [batch, dim]
        
        # a) Project momentum onto static basis
        proj_static = torch.matmul(v, self.U) # [batch, rank]
        
        # b) Modulate projection by Context (Hyper-U)
        proj_dynamic = proj_static * g_u # [batch, rank]
        
        # c) Soft-Saturation (to prevent energy explosion)
        # Instead of pure quadratic sq_dynamic = proj_dynamic * proj_dynamic
        sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))
        
        # d) Modulate Reconstruction by Context (Hyper-W)
        sq_modulated = sq_dynamic * g_w # [batch, rank]
        
        # e) Reconstruct force
        # out = sq_modulated @ W.T
        out = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]
        
        # 3. Apply inherited Active Inference (Plasticity/Singularities)
        # Note: HyperChristoffel currently inherits from LowRankChristoffel directly.
        # Active inference features from ReactiveChristoffel are not automatically included unless explicitly mixed in.
        
        return torch.clamp(out, -self.clamp_val, self.clamp_val)
