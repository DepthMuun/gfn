"""
gfn/models/components/aggregators/momentum.py — GFN V5
Portado desde: gfn_old/nn/layers/aggregation/momentum_aggregator.py

MomentumAggregator: EMA de estados con acumulación de trayectoria.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class MomentumAggregator(nn.Module):
    """
    Aggregator basado en momentum de trayectoria.

    Soporta:
      - EMA del estado corriente (running state)
      - Acumulación completa de trayectoria (sum o avg sobre L)

    Útil para incorporar contexto acumulado a lo largo de una secuencia.
    """

    def __init__(self, dim: int, momentum: float = 0.9, alpha: float = 0.1,
                 mode: str = 'avg'):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.alpha = alpha
        self.mode = mode
        self.running_state: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Limpia el estado acumulado (usar al inicio de cada secuencia nueva)."""
        self.running_state = None

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, 
                reset: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: Secuencia completa [B, L, D] para acumulación
            v_seq: Secuencia completa de velocidades [B, L, D]
            reset: Si True, reinicia el estado acumulado (EMA opcional)
        Returns:
            (x_agg, v_agg, weights)
        """
        B, L, D = x_seq.shape
        
        # 1. Acumulación de trayectoria
        accumulated = x_seq.sum(dim=1) if self.mode == 'sum' else x_seq.mean(dim=1)
        base = x_seq[:, -1]
        x_agg = base + self.alpha * accumulated
        
        # 2. EMA Update (opcional si se llama paso a paso, pero el plugin lo usa en batch)
        if reset or self.running_state is None or self.running_state.shape != x_agg.shape:
            self.running_state = x_agg.clone()
        else:
            self.running_state = (
                self.momentum * self.running_state + (1.0 - self.momentum) * x_agg
            )

        weights = torch.ones(B, L, device=x_seq.device) / L
        return x_agg, x_agg, weights
