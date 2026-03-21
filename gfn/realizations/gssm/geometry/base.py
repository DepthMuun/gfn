import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ..interfaces.geometry import Geometry
from ..config.schema import PhysicsConfig
from ..constants import TOPOLOGY_EUCLIDEAN

class BaseGeometry(nn.Module):
    """
    Base implementation for Riemannian Geometries in GFN V5.
    Conforms to the Geometry protocol.
    """
    def __init__(self, config: Optional[PhysicsConfig] = None):
        super().__init__()
        self.config = config or PhysicsConfig()
        self.return_friction_separately = True
        self.topology_type = self.config.topology.type
        
    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Default metric is identity (Euclidean). Subclasses should override."""
        return torch.ones_like(x)
        
    def christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        """Default Christoffel symbols are zero (Euclidean). Subclasses should override."""
        return torch.zeros_like(x)

    def compute_kinetic_energy(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Calculates Riemannian kinetic energy: T = (1/2) Σ_i g_ii v_i²
        Supports position-dependent metrics (like Torus).
        """
        g = self.metric_tensor(x)  # [..., D]
        return 0.5 * (g * v.pow(2)).sum(dim=-1)

    def compute_potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates physical potential energy V(x). 
        Default is 0.0 unless overwritten by specific topologies or forces.
        """
        return torch.zeros_like(x).sum(dim=-1)
        
    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, force: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes acceleration: acc = -Gamma(v, v) + F/g
        Subclasses can override for more complex physics.
        """
        if v is None:
            return torch.zeros_like(x)
            
        gamma = self.christoffel_symbols(x)
        # Standard geodesic acceleration: -Gamma^k_ij v^i v^j
        # In our simplified 1D-per-dimension metric, it's often just a point-wise product
        acc = -gamma * (v**2) 
        
        if force is not None:
            g = self.metric_tensor(x)
            acc = acc + (force / (g + 1e-8))
            
        if getattr(self, 'return_friction_separately', False):
            return acc, torch.zeros_like(v) if v is not None else torch.zeros_like(x)
            
        return acc

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Default projection is identity. Subclasses should override for periodic spaces."""
        return x

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Default distance is Euclidean. Subclasses should override."""
        return torch.norm(x1 - x2, dim=-1)
