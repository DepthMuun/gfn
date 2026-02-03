import torch
import torch.nn as nn
from ..constants import (
    CURVATURE_CLAMP, FRICTION_SCALE, DEFAULT_FRICTION,
    EPSILON_STRONG, GATE_BIAS_OPEN
)

try:
    from gfn.cuda.ops import christoffel_fused,CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

class LowRankChristoffel(nn.Module):
    r"""
    Computes the Christoffel symbols \Gamma^k_{ij} using a low-rank decomposition.
    To ensure symmetry in lower indices (torsion-free), we use a symmetric decomposition:
    \Gamma^k_{ij} = \sum_{r=1}^R \lambda_{kr} * (U_{ir} * U_{jr})
    
    Args:
        dim (int): Dimension of the manifold (hidden size).
        rank (int): Rank of the decomposition.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.config = physics_config or {}
        self.clamp_val = self.config.get('stability', {}).get('curvature_clamp', CURVATURE_CLAMP)
        self.is_torus = self.config.get('topology', {}).get('type', '').lower() == 'torus'
        
        # Toroidal gates use Fourier features [sin(x), cos(x)]
        gate_input_dim = 2 * dim if self.is_torus else dim
        
        # Initialize U/W to start flat
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.W = nn.Parameter(torch.zeros(dim, rank))
        
        # Friction coefficient for Conformal Symplectic System
        self.friction = self.config.get('stability', {}).get('friction', DEFAULT_FRICTION)
        
        # Position gate for potential strength, initialized near zero
        self.V = nn.Linear(gate_input_dim, 1, bias=False)
        nn.init.zeros_(self.V.weight)
        
        # Adaptive curvature gate
        self.gate_proj = nn.Linear(gate_input_dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, GATE_BIAS_OPEN)  # Start OPEN (sigmoid(2) ~ 0.88)
        
        # State component of friction gate
        self.forget_gate = nn.Linear(gate_input_dim, dim)
        nn.init.normal_(self.forget_gate.weight, std=0.01)
        
        # Force component of friction gate
        self.input_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.input_gate.weight, std=0.01)
        
        nn.init.constant_(self.forget_gate.bias, 0.0) 
        
    def forward(self, v, x=None, force=None, **kwargs):
        """
        Compute Generalized Force: Γ(v, v) + Friction(x)*v
        
        Output represents the effective "Resistance" to motion.
        Acc = F_ext - Output
        """
        # Use fused CUDA kernel when available
        try:
            from gfn.cuda.ops import lowrank_christoffel_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and v.is_cuda and v.dim() == 2:
                x_empty = torch.empty(0, device=v.device)
                V_empty = torch.empty(0, device=v.device)
                gamma_cuda = lowrank_christoffel_fused(v, self.U, self.W, x_empty, V_empty, 0.0, 1.0, 1.0)
                
                if x is not None:
                     if self.is_torus:
                         x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
                     else:
                         x_in = x

                     friction = torch.sigmoid(self.forget_gate(x_in)) * FRICTION_SCALE
                     
                     if getattr(self, 'return_friction_separately', False):
                         return gamma_cuda, friction
                         
                     gamma_cuda = gamma_cuda + friction * v
                return torch.clamp(gamma_cuda, -self.clamp_val, self.clamp_val)
        except Exception as e:
            print(f"[GFN:WARN] CUDA lowrank_christoffel failed: {e}, falling back to PyTorch")
            # Fall through to PyTorch implementation
    
        # PyTorch fallback with improved numerical stability
        if v.dim() == 3 and self.U.dim() == 3:
            proj = torch.bmm(v, self.U) 
            norm = torch.norm(proj, dim=-1, keepdim=True)
            # Double epsilon protection for division
            scale = 1.0 / (1.0 + norm + 1e-4)
            sq = (proj * proj) * scale 
            gamma = torch.bmm(sq, self.W.transpose(1, 2)) 
        else:
            proj = torch.matmul(v, self.U)
            norm = torch.norm(proj, dim=-1, keepdim=True)
            # Double epsilon protection for division
            scale = 1.0 / (1.0 + norm + 1e-4)
            sq = (proj * proj) * scale
            gamma = torch.matmul(sq, self.W.t())
            
        if x is not None:
            if self.is_torus:
                 x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            else:
                 x_in = x
                 
            Wf = kwargs.get('W_forget_stack', None)
            Wi = kwargs.get('W_input_stack', None)
            bf = kwargs.get('b_forget_stack', None)
            
            if Wf is not None and bf is not None:
                if Wf.dim() == 3: Wf = Wf[0] # Handle Depth head
                if Wi is not None and Wi.dim() == 3: Wi = Wi[0]
                if bf.dim() == 2: bf = bf[0]
                
                gate_activ = torch.matmul(x_in, Wf.t()) + bf
                if Wi is not None and force is not None:
                     gate_activ = gate_activ + torch.matmul(force, Wi.t())
            else:
                gate_activ = self.forget_gate(x_in)
                if force is not None:
                    gate_activ = gate_activ + self.input_gate(force)
                
            mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
            
            if getattr(self, 'return_friction_separately', False):
                 gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
                 return gamma, mu
                 
            gamma = gamma + mu * v
            
        return CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
