import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class ResidualDynamics(BaseDynamics):
    """
    Residual (Skip-Connection) Dynamics.
    
    Implements a true residual connection:
    state_next = current_state + scale * norm(proposal - current_state)
    
    This allows the model to learn a correction over the current state
    instead of completely replacing it (like direct) or interpolating it (like mix).
    
    For POSITION (torus): wrapping to [-π, π] after the residual.
    For VELOCITY (euclidean): RMSNorm over the residual.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN, 
                 residual_scale: float = 0.1, **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        # Residual scale - learnable parameter but initially small
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Calculate residual: difference between proposal and current state
        if self.topology == TOPOLOGY_TORUS:
            # Geodesic difference on the torus
            residual = torch.atan2(torch.sin(absolute_proposal - current_state),
                                   torch.cos(absolute_proposal - current_state))
        else:
            residual = absolute_proposal - current_state
        
        # Apply normalization to residual
        # Note: for velocities, context_x (position) enables MetricNormalization
        residual_normalized = self._apply_norm(residual, context_x=context_x)
        
        # Scale residual with learnable parameter
        scale = torch.sigmoid(self.residual_scale)
        
        # Apply residual connection: state + scale * residual
        next_state = current_state + scale * residual_normalized
        
        # On the torus, ensure we stay in [-π, π]
        if self.topology == TOPOLOGY_TORUS:
            next_state = torch.atan2(torch.sin(next_state), torch.cos(next_state))
            
        return next_state
