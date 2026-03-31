"""
gfn/physics/gating.py — GFN V5
Ported from: gfn_old/nn/layers/physics/gating/core.py
              gfn_old/nn/layers/physics/thermo.py

Gating modules for dynamic time (adaptive dt per head).
"""
import torch
import torch.nn as nn
from typing import Optional
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN


class RiemannianGating(nn.Module):
    """
    Riemannian curvature gating.
    
    If manifold curvature is high → small dt (complex region).
    If curvature is low (flat) → large dt (skip-connection behavior).
    
    For toroidal topology: input expands to [sin(x), cos(x)]
    to maintain circular phase invariance.
    """
    def __init__(self, dim: int, topology: str = TOPOLOGY_EUCLIDEAN):
        super().__init__()
        self.topology = str(topology).lower().strip()
        input_dim = 2 * dim if self.topology == TOPOLOGY_TORUS else dim
        
        self.curvature_net = nn.Sequential(
            nn.Linear(input_dim, max(1, dim // 4)),
            nn.Tanh(),
            nn.Linear(max(1, dim // 4), 1),
            nn.Sigmoid()
        )
        
        # Initialization: start with gate ~0.88 (open, bias=2.0)
        with torch.no_grad():
            nn.init.constant_(self.curvature_net[2].bias, 2.0)
            nn.init.xavier_uniform_(self.curvature_net[0].weight, gain=0.1)
            nn.init.xavier_uniform_(self.curvature_net[2].weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, head_dim] — position state for a head
        Returns:
            gate: [B, 1] — scaling factor for dt in [0, 1]
        """
        if self.topology == TOPOLOGY_TORUS:
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.curvature_net(x)


class ThermodynamicLayer(nn.Module):
    """
    Thermodynamic Gating (Paper 03).
    Modulates dt based on Hamiltonian energy: H = T + U.
    
    If energy is above reference → small dt (hot system).
    If energy is below → larger dt (cold system).
    """
    def __init__(self, dim: int, temperature: float = 1.0,
                 ref_energy: float = 1.0, sensitivity: float = 1.0):
        super().__init__()
        self.dim = dim
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
        self.ref_H = nn.Parameter(torch.tensor(ref_energy))
        self.sensitivity = sensitivity

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, head_dim] — position
            v: [B, head_dim] — velocity
        Returns:
            gate: [B, 1] — thermodynamic scaling factor
        """
        K = 0.5 * (v ** 2).sum(dim=-1, keepdim=True)   # Kinetic energy
        U = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)   # Potential energy (approx.)
        H = K + U
        T = self.log_temp.exp()
        logits = (self.ref_H - H) / (T * self.sensitivity)
        return torch.sigmoid(logits)


class FrictionGate(nn.Module):
    """
    Friction Gate with position and force dependency (Paper 25).
    
    mu(x, f) = sigmoid(W_f * x + W_i * f + b) * FRICTION_SCALE
    
    Supports single-head and multi-head modes.
    """
    def __init__(self, dim: int, gate_input_dim: Optional[int] = None, mode: str = 'mlp'):
        super().__init__()
        from ..constants import FRICTION_SCALE, GATE_BIAS_CLOSED 
        self.dim = dim
        self.input_dim = gate_input_dim if gate_input_dim is not None else dim
        self.friction_scale = FRICTION_SCALE
        self.mode = mode

        if mode == 'mlp':
            self.forget_gate = nn.Linear(self.input_dim, dim)
            self.input_gate = nn.Linear(dim, dim, bias=False)
            nn.init.normal_(self.forget_gate.weight, std=0.01)
            nn.init.constant_(self.forget_gate.bias, GATE_BIAS_CLOSED)
            nn.init.normal_(self.input_gate.weight, std=0.01)
        else:
            self.forget_gate = None
            self.input_gate = None

    def forward(self, x: torch.Tensor, force: Optional[torch.Tensor] = None,
                v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            mu: [B, dim] — position-dependent friction coefficient
        """
        if self.mode != 'mlp' or self.forget_gate is None:
            return torch.zeros(*x.shape[:-1], self.dim, device=x.device, dtype=x.dtype)

        gate_activ = self.forget_gate(x)
        if force is not None:
            gate_activ = gate_activ + self.input_gate(force)

        mu = torch.sigmoid(gate_activ) * self.friction_scale
        
        if v is not None:
            from ..constants import DEFAULT_FRICTION
            v_norm = torch.norm(v, dim=-1, keepdim=True)
            mu = mu * (1.0 + DEFAULT_FRICTION * v_norm)

        return mu


__all__ = ['RiemannianGating', 'ThermodynamicLayer', 'FrictionGate']
