import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class StochasticDynamics(BaseDynamics):
    """
    Stochastic Dynamics: state_next = norm(proposal + sigma * epsilon).
    
    Adds learnable noise for exploration in geometric GFNs.
    Ideal for preventing collapse into local geodesics during training.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN,
                 sigma_init: float = 0.01, mode: str = 'residual', **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        self.sigma = nn.Parameter(torch.tensor(sigma_init))
        self.mode = mode

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if self.mode == 'residual':
            base = current_state + (absolute_proposal - current_state)
        else:
            base = absolute_proposal

        # sigma always positive; noise at learnable scale
        sigma = torch.nn.functional.softplus(self.sigma) + 1e-6
        noise = torch.randn_like(base) * sigma
        return self._apply_norm(base + noise, context_x=context_x)
