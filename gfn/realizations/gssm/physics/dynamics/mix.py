import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class MixDynamics(BaseDynamics):
    """
    Mix Dynamics: state_next = norm(alpha * current + (1 - alpha) * proposal).
    
    alpha is a learnable parameter that controls the memory of the previous state.
    We use log_alpha to avoid saturation and allow exploration of the interpolation space.
    
    Recommended initialization: alpha close to 0.0 to give more weight to the initial proposal,
    then the model gradually learns the optimal balance.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN,
                 alpha_init: float = 0.3, **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        # Use log_alpha to avoid sigmoid saturation
        # alpha = sigmoid(log_alpha) -> full range (0, 1)
        self.log_alpha = nn.Parameter(torch.tensor([alpha_init]))
        
        # Change scale for stability
        self.change_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Convert log_alpha to alpha in (0, 1)
        alpha = torch.sigmoid(self.log_alpha)
        
        # Interpolation between current state and proposal
        if self.topology == TOPOLOGY_TORUS:
            # Geodesic Interpolation (Circular Slerp)
            # We average in the embedding space (sin, cos) and return to angle
            interpolated = torch.atan2(
                alpha * torch.sin(current_state) + (1.0 - alpha) * torch.sin(absolute_proposal),
                alpha * torch.cos(current_state) + (1.0 - alpha) * torch.cos(absolute_proposal)
            )
        else:
            # Standard Euclidean interpolation
            interpolated = alpha * current_state + (1.0 - alpha) * absolute_proposal
        
        # Apply normalization according to topology (context_x allows metric-aware)
        result = self._apply_norm(interpolated, context_x=context_x)
        
        # Apply change scale (soft learning rate)
        if self.topology == TOPOLOGY_TORUS:
            # On the torus, the difference is also circular
            diff = torch.atan2(torch.sin(result - current_state), torch.cos(result - current_state))
            result = current_state + self.change_scale * diff
        else:
            result = current_state + self.change_scale * (result - current_state)
        
        return result
    
    def get_alpha(self) -> float:
        """Returns the current alpha value for debugging."""
        return float(torch.sigmoid(self.log_alpha).item())
