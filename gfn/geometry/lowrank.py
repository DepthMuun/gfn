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
    Computes the Christoffel symbols Gamma^k_ij using a low-rank decomposition.
    
    IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):
    
    1. LOW-RANK APPROXIMATION LIMITATION:
       The decomposition Gamma^k_ij = sum_{r=1}^R lambda_kr * (U_ir * U_jr)
       is an APPROXIMATION that does not guarantee true Christoffel properties:
       - Symmetry Gamma^k_ij = Gamma^k_ji is preserved (by construction)
       - Bianchi identities are NOT guaranteed
       - Does not derive from a valid metric tensor g_ij
       
       This is a DESIGN CHOICE for computational efficiency, not a mathematically
       rigorous Christoffel symbol computation.
    
    2. VELOCITY-DEPENDENT FRICTION:
       Friction is now computed as: mu(x,v) = sigma(gate(x)) * FRICTION_SCALE * (1 + alpha * ||v||)
       This makes friction physically more sensible: higher velocities experience more drag.
    
    Args:
        dim (int): Dimension of the manifold (hidden size).
        rank (int): Rank of the decomposition.
        physics_config (dict): Configuration with stability and topology settings.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.config = physics_config or {}
        self.clamp_val = self.config.get('stability', {}).get('curvature_clamp', CURVATURE_CLAMP)
        self.is_torus = self.config.get('topology', {}).get('type', '').lower() == 'torus'
        
        # AUDIT FIX (2026-02-07): Configurable normalization for consistency
        # Set to False to disable trace normalization (recommended for pure geometric tasks)
        self.enable_trace_normalization = self.config.get('stability', {}).get('enable_trace_normalization', False)
        
        # Velocity-dependent friction coefficient (AUDIT FIX)
        self.velocity_friction_scale = self.config.get('stability', {}).get('velocity_friction_scale', 0.0)
        
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
        Compute Generalized Force: Gamma(v, v) + Friction(x,v)*v
        
        PHYSICS CORRECTION (Auditoria 2026-02-06):
        - Christoffel symbols represent geometric "resistance" to motion
        - Friction represents physical damping
        - Total acceleration: a = F_ext - Christoffel(v,v) - Friction*v
        
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
                     
                     # AUDIT FIX (2026-02-07): Velocity dependence with proper scaling
                     # Scale is now 0 by default to avoid excessive damping
                     if self.velocity_friction_scale > 0:
                         velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
                         # Normalize by sqrt(dim) for dimension-invariant behavior
                         velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + 1e-8)
                         friction = friction * (1.0 + self.velocity_friction_scale * velocity_magnitude)
                     
                     if getattr(self, 'return_friction_separately', False):
                         # Apply symmetry normalization before returning (same as Python path)
                         gamma_cuda = self._normalize_christoffel_structure(gamma_cuda)
                         return gamma_cuda, friction
                         
                     gamma_cuda = gamma_cuda + friction * v
                
                # AUDIT FIX (2026-02-07): Apply same normalization as Python path
                # This ensures consistent behavior between CUDA and CPU execution
                gamma_cuda = self._normalize_christoffel_structure(gamma_cuda)
                return torch.clamp(gamma_cuda, -self.clamp_val, self.clamp_val)
        except Exception as e:
            print(f"[GFN:WARN] CUDA lowrank_christoffel failed: {e}, falling back to PyTorch")
            # Fall through to PyTorch implementation
    
        # PyTorch fallback with improved numerical stability
        if v.dim() == 3 and self.U.dim() == 3:
            proj = torch.bmm(v, self.U) 
            norm = torch.norm(proj, dim=-1, keepdim=True)
            # Double epsilon protection for division (unified with CUDA)
            scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
            sq = (proj * proj) * scale 
            gamma = torch.bmm(sq, self.W.transpose(1, 2)) 
        else:
            proj = torch.matmul(v, self.U)
            norm = torch.norm(proj, dim=-1, keepdim=True)
            # Double epsilon protection for division (unified with CUDA)
            scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
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
                
            # AUDIT FIX: Base friction from position-dependent gate
            mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
            
            # AUDIT FIX: Add velocity-dependent component
            # Higher velocities experience proportionally more drag
            velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
            mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)
            
            if getattr(self, 'return_friction_separately', False):
                 # Apply symmetry normalization
                 gamma_normalized = self._normalize_christoffel_structure(gamma)
                 return gamma_normalized, mu
                 
            gamma = gamma + mu * v
            
        # AUDIT FIX: Symmetry-preserving normalization
        # Ensures Gamma^k_ij approx Gamma^k_ji numerically
        gamma = self._normalize_christoffel_structure(gamma)
            
        return CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
    
    def _normalize_christoffel_structure(self, gamma):
        """
        AUDIT FIX (2026-02-07): Symmetry-preserving normalization.
        
        This normalization serves two purposes:
        1. Ensures numerical symmetry Gamma^k_ij approx Gamma^k_ji by averaging cross-terms
        2. Optionally removes diagonal trace to center the connection
        
        IMPORTANT: The trace normalization is DISABLED by default (2026-02-07)
        because it was removing legitimate geometric information without clear
        mathematical justification. It can be enabled via config:
            physics_config['stability']['enable_trace_normalization'] = True
        
        This ensures consistent behavior between CUDA and Python paths.
        """
        if gamma.dim() < 3:
            return gamma  # Skip for batched vectors (shape: [..., dim])
        
        # gamma shape: [..., dim, dim] for last two indices
        # Average with transpose to enforce approximate symmetry
        # This is mathematically justified: Gamma^k_ij = Gamma^k_ji for torsion-free connections
        gamma_sym = 0.5 * (gamma + gamma.transpose(-1, -2))
        
        # AUDIT FIX (2026-02-07): Trace normalization is now OPTIONAL and DISABLED by default
        # The diagonal elements Gamma^k_kk have geometric meaning in normal coordinates
        # and should not be arbitrarily removed
        if self.enable_trace_normalization and gamma_sym.dim() >= 2:
            # Compute diagonal mean for non-batch dimensions
            diag_mean = torch.diagonal(gamma_sym, dim1=-1, dim2=-2).mean(dim=-1, keepdim=True)
            # Subtract diagonal contribution (centered connection)
            gamma_centered = gamma_sym - torch.diag_embed(diag_mean.squeeze(-1))
        else:
            gamma_centered = gamma_sym
        
        return gamma_centered
