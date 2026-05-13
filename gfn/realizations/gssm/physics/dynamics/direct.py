import torch
from typing import Optional
from .base import BaseDynamics


class DirectDynamics(BaseDynamics):
    """
    Direct (Geodesic) Dynamics: state_next = norm(proposal).
    El flujo geodésico directo — la propuesta del integrador es el resultado.
    """
    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self._apply_norm(absolute_proposal, context_x=context_x)
