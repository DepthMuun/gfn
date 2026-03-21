"""
gfn/physics/dynamics/base.py — GFN V5
Base abstracta para los módulos de Dynamics.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Any
from ...physics.normalization import IdentityNormalization


class BaseDynamics(nn.Module, ABC):
    """
    Clase base abstracta para los módulos de actualización de estado del manifold.

    Contrato:
      forward(current_state, absolute_proposal) → state_next

    topology='torus':     wrapping isométrico en [-π, π]
    topology='euclidean': aplica norm_layer (generalmente RMSNorm o Identity)
    """

    def __init__(self, dim: int, norm_layer: Optional[nn.Module] = None,
                 topology: str = 'euclidean', **kwargs):
        super().__init__()
        self.dim = dim
        self.topology = topology.lower()
        self.norm = norm_layer if norm_layer is not None else IdentityNormalization()

    def _apply_norm(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aplicación de la normalización geométrica inyectada por ManifoldLayer."""
        if self.topology == 'torus':
            # Wrapping isométrico: preserva invarianza de fase circular
            return torch.atan2(torch.sin(x), torch.cos(x))
        try:
            return self.norm(x, context_x=context_x)
        except TypeError:
            # Fallback for layers that don't accept context_x (e.g. standard nn.Identity)
            return self.norm(x)

    @abstractmethod
    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
