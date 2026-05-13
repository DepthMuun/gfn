"""
core/state.py — GFN V5
Manejo de estado del manifold (posición + velocidad).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ManifoldStateManager:
    """
    Gestiona la inicialización y manipulación del estado (x, v).
    Compatible con batches y múltiples cabezales.
    """

    @staticmethod
    def initialize(x0: nn.Parameter, v0: nn.Parameter,
                   batch_size: int, n_trajectories: int = 1,
                   initial_spread: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inicializa el estado (x, v) para un batch dado.

        Args:
            x0, v0:         Parámetros iniciales [1, H, HD]
            batch_size:     Tamaño del batch
            n_trajectories: Número de trayectorias paralelas
            initial_spread: Ruido inicial

        Returns:
            (x, v) — [B, H, HD]
        """
        x = x0.expand(batch_size, -1, -1)
        v = v0.expand(batch_size, -1, -1)

        if initial_spread > 0:
            x = x + torch.randn_like(x) * initial_spread

        return x.contiguous(), v.contiguous()

    @staticmethod
    def from_tuple(state: Optional[Tuple], x0: nn.Parameter, v0: nn.Parameter,
                   batch_size: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construye (x, v) desde un estado previo o desde parámetros iniciales.
        Compatible con el API de BasicModel.
        """
        if state is not None and isinstance(state, (tuple, list)) and len(state) == 2:
            return state[0], state[1]
        return ManifoldStateManager.initialize(x0, v0, batch_size, **kwargs)

    @staticmethod
    def wrap_torus(x: torch.Tensor) -> torch.Tensor:
        """Proyecta posición al dominio toroidal [-π, π]."""
        return torch.atan2(torch.sin(x), torch.cos(x))

    @staticmethod
    def energy(v: torch.Tensor) -> torch.Tensor:
        """Energía cinética H = 0.5 * ||v||² por muestra."""
        return 0.5 * (v ** 2).sum(dim=-1)
