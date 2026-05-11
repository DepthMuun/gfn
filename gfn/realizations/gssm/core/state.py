"""
core/state.py — GFN V5
Manifold state management (position + velocity).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ManifoldStateManager:
    """
    Manages initialization and manipulation of state (x, v).
    Compatible with batches and multiple heads.
    """

    @staticmethod
    def initialize(x0: nn.Parameter, v0: nn.Parameter,
                   batch_size: int, n_trajectories: int = 1,
                   initial_spread: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the state (x, v) for a given batch.

        Args:
            x0, v0:         Initial parameters [1, H, HD]
            batch_size:     Batch size
            n_trajectories: Number of parallel trajectories
            initial_spread: Initial noise

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
        Builds (x, v) from a previous state or from initial parameters.
        Compatible with BasicModel API.
        """
        if state is not None and isinstance(state, (tuple, list)) and len(state) == 2:
            return state[0], state[1]
        return ManifoldStateManager.initialize(x0, v0, batch_size, **kwargs)

    @staticmethod
    def wrap_torus(x: torch.Tensor) -> torch.Tensor:
        """Projects position to toroidal domain [-π, π]."""
        return torch.atan2(torch.sin(x), torch.cos(x))

    @staticmethod
    def energy(v: torch.Tensor) -> torch.Tensor:
        """Kinetic energy H = 0.5 * ||v||² per sample."""
        return 0.5 * (v ** 2).sum(dim=-1)
