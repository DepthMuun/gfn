"""
gfn/models/components/aggregators/pooling.py — GFN V5
Portado desde: gfn_old/nn/layers/aggregation/pooling_aggregator.py

HamiltonianPooling: agrega secuencias de estados ponderando por energía Hamiltoniana.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ....constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN


class HamiltonianPooling(nn.Module):
    """
    Aggregation ponderado por energía Hamiltoniana.

    H = K + U donde:
      K = 0.5 * v^T * g * v  (energía cinética — diagonal con métrica aprendible)
      U = 0.5 * ||x||²       (energía potencial cuadrática)

    Estados con mayor H obtienen mayor peso de atención.
    
    Motivación física:
      - Los estados de alta energía son más "importantes" en sistemas dinámicos.
      - La energía indica la fuerza de interacción/aplicación de fuerza.
    """

    def __init__(self, dim: int, temperature: float = 1.0, 
                 learn_metric: bool = False, topology_type: str = TOPOLOGY_EUCLIDEAN):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.learn_metric = learn_metric
        self.topology_type = topology_type.lower().strip()
        
        # TOROIDAL_PERIOD comes from constants if needed, but 2*pi is standard
        self.period = 2.0 * 3.14159265358979

        if learn_metric:
            self.metric = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('metric', torch.ones(dim))

    def kinetic_energy(self, v: torch.Tensor) -> torch.Tensor:
        """K = 0.5 * v^T g v. v: [B, L, D] → [B, L]"""
        B, L, D = v.shape
        metric_expanded = self.metric.view(1, 1, -1).expand(B, L, -1)
        weighted_v = v * metric_expanded
        return 0.5 * (v * weighted_v).sum(dim=-1)

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """U = potential. x: [B, L, D] → [B, L]"""
        if self.topology_type == TOPOLOGY_TORUS:
            # Soft-Potential for Torus: 1 - cos(x)
            # This is periodic and centered at 0
            return (1.0 - torch.cos(x)).sum(dim=-1)
        return 0.5 * (x ** 2).sum(dim=-1)

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: Secuencia de posiciones [B, L, D]
            v_seq: Secuencia de velocidades [B, L, D]
        Returns:
            (x_agg [B, D], v_agg [B, D], weights [B, L])
        """
        K = self.kinetic_energy(v_seq)
        U = self.potential_energy(x_seq)
        H = K + U  # [B, L]

        weights = F.softmax(H / self.temperature, dim=-1)  # [B, L]
        x_agg = (weights.unsqueeze(-1) * x_seq).sum(dim=1)
        v_agg = (weights.unsqueeze(-1) * v_seq).sum(dim=1)

        return x_agg, v_agg, weights
