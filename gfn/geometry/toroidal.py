import torch
import torch.nn as nn
from ..constants import (
    TOROIDAL_MAJOR_RADIUS, TOROIDAL_MINOR_RADIUS,
    TOROIDAL_CURVATURE_SCALE, FRICTION_SCALE,
    EPSILON_SMOOTH, CLAMP_MIN_STRONG, GATE_BIAS_CLOSED,
    DEFAULT_PLASTICITY, SINGULARITY_THRESHOLD, BLACK_HOLE_STRENGTH
)

class ToroidalChristoffel(nn.Module):
    """
    Toroidal Geometry (Curved Torus).
    
    Metric for a 2D torus in 3D is:
    g = diag(r^2, (R + r cos th)^2)
    
    We generalize this to N dimensions by alternating 'inner' and 'outer' coords,
    or simply providing a periodic metric that isn't identity.
    
    Structure:
    - Theta: Local circle coordinate
    - Phi: Global circle coordinate
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.config = physics_config or {}
        # Toroidal parameters: R (major radius), r (minor radius)
        self.R = self.config.get('topology', {}).get('major_radius', TOROIDAL_MAJOR_RADIUS)
        self.r = self.config.get('topology', {}).get('minor_radius', TOROIDAL_MINOR_RADIUS)
        
        self.topology_id = 1
        self.is_torus = True
        
        # Friction Gates (The Clutch)
        # Input to gates: [batch, 2*dim] (sin(x), cos(x))
        gate_input_dim = 2 * dim
        
        # State component of friction gate
        self.forget_gate = nn.Linear(gate_input_dim, dim)
        nn.init.normal_(self.forget_gate.weight, std=0.01)
        nn.init.constant_(self.forget_gate.bias, GATE_BIAS_CLOSED)  # Release the clutch (Low friction start) 
        
        # Force component of friction gate
        self.input_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.input_gate.weight, std=0.01)

        self.clamp_val = self.config.get('stability', {}).get('curvature_clamp', 5.0)
        
        # ACTIVE INFERENCE (Restored for Torus)
        self.active_cfg = self.config.get('active_inference', {})
        self.plasticity = self.active_cfg.get('reactive_curvature', {}).get('plasticity', DEFAULT_PLASTICITY)
        self.singularity_threshold = self.active_cfg.get('singularities', {}).get('threshold', SINGULARITY_THRESHOLD)
        self.black_hole_strength = self.active_cfg.get('singularities', {}).get('strength', BLACK_HOLE_STRENGTH)
        
        # Potential Gate for Singularities
        self.V = nn.Linear(gate_input_dim, 1) if self.active_cfg.get('singularities', {}).get('enabled', False) else None
        if self.V:
             nn.init.constant_(self.V.bias, GATE_BIAS_CLOSED)  # Start with no singularities
        
    def get_metric(self, x):
        """
        Returns diagonal metric tensor g_ii(x).
        g_theta = r^2
        g_phi = (R + r cos theta)^2
        """
        g = torch.ones_like(x)
        for i in range(0, self.dim - 1, 2):
            th = x[..., i]
            # g[..., i] = r^2
            g[..., i] = self.r**2
            # g[..., i+1] = (R + r cos th)^2
            g[..., i+1] = (self.R + self.r * torch.cos(th))**2
        return g

    def forward(self, v, x=None, force=None, **kwargs):
        """
        Computed Christoffel Force: Gamma(v, v)^k = Gamma^k_ij v^i v^j
        """
        if x is None: return torch.zeros_like(v)
        
        # We assume coordinates come in pairs (th, ph) or are all S1.
        # Let's implement a 'Chain Torus' where th_i affects ph_i.
        
        # 1. Coordinate parity
        # For even-dim, half are theta (inner), half are phi (outer)
        # For odd-dim, we pad or alternate.
        
        # We'll use a simple tiling: x[0] affects x[1], x[1] affects x[2]... 
        # But for symmetry, we'll use: 
        # x_even = theta (inner)
        # x_odd = phi (outer)
        
        # Precompute cos/sin of theta
        cos_th = torch.cos(x)
        sin_th = torch.sin(x)
        
        gamma = torch.zeros_like(v)
        
        # N-dimensional torus logic: 
        # For each pair (i, i+1):
        # Gamma^i_{i+1, i+1} = (R + r cos x_i) sin x_i / r
        # Gamma^{i+1}_{i+1, i} = - (r sin x_i) / (R + r cos x_i)
        
        for i in range(0, self.dim - 1, 2):
            th = x[..., i]
            v_th = v[..., i]
            v_ph = v[..., i+1]
            
            # Double protection against division by zero
            # 1. Clamp the denominator to prevent near-zero values
            denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)
            
            # 2. Add epsilon in division for extra safety
            term_th = denom * torch.sin(th) / (self.r + EPSILON_SMOOTH)
            gamma[..., i] = term_th * (v_ph ** 2)
            
            # Double epsilon protection for the second term
            term_ph = -(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)
            gamma[..., i+1] = 2.0 * term_ph * v_ph * v_th
            
        gamma = gamma * TOROIDAL_CURVATURE_SCALE  # Strong Curvature (User Requested Full Torus)
        
        # APPLY THE CLUTCH (DYNAMIC FRICTION)
        # Map to Periodic Space: [sin(x), cos(x)]
        x_in = torch.cat([sin_th, cos_th], dim=-1)
             
        # Base friction from state
        gate_activ = self.forget_gate(x_in)
        
        if force is not None:
             gate_activ = gate_activ + self.input_gate(force)
            
        # Level 34: BRAKING POWER (STIFFNESS)
        # Higher mu to stop v=~15 in one step at dt=0.2
        mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
        
        # ACTIVE INFERENCE LOGIC (Triad v2.0)
        if self.active_cfg.get('enabled', False):
             # 1. Reactive Curvature (Plasticity)
             if self.active_cfg.get('reactive_curvature', {}).get('enabled', False):
                  energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
                  gamma = gamma * (1.0 + self.plasticity * energy)
                  
             # 2. Logical Singularities (Black Holes)
             if self.active_cfg.get('singularities', {}).get('enabled', False) and self.V is not None:
                  potential = torch.sigmoid(self.V(x_in))
                  is_singularity = (potential > self.singularity_threshold).float()
                  gamma = gamma * (1.0 + is_singularity * (self.black_hole_strength - 1.0))

        # If we want to return BOTH for implicit integration
        if getattr(self, 'return_friction_separately', False):
             return gamma, mu
             
        gamma = gamma + mu * v
        return gamma
