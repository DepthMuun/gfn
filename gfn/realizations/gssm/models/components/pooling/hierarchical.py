"""
gfn/models/components/aggregators/hierarchical.py — GFN V5
Portado desde: gfn_old/nn/layers/aggregation/hierarchical_aggregator.py

HierarchicalAggregator: pooling jerárquico (local → global) via Hamiltonian Pooling.
"""
import torch
import torch.nn as nn
from typing import Tuple
from ....constants import TOPOLOGY_EUCLIDEAN
from .pooling import HamiltonianPooling


class HierarchicalAggregator(nn.Module):
    """
    Aggregator Jerárquico Multi-escala.

    Realiza pooling Hamiltoniano local por bloques, seguido de aggregación global.
    Motivación física: los sistemas de muchos cuerpos tienen organización jerárquica
    (clusters → grupos → sistema completo).
    """

    def __init__(self, dim: int, temperature: float = 1.0, topology_type: str = TOPOLOGY_EUCLIDEAN):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.local_pool = HamiltonianPooling(dim, temperature=temperature, topology_type=topology_type)
        self.global_pool = HamiltonianPooling(dim, temperature=temperature, topology_type=topology_type)

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: [B, L, D]
            v_seq: [B, L, D]
        Returns:
            (x_agg [B, D], v_agg [B, D], local_weights [B * n_blocks, block_size])
        """
        B, L, D = x_seq.shape
        block_size = max(2, L // 4)

        # Padding para divisibilidad exacta por block_size
        pad_len = (block_size - (L % block_size)) % block_size
        if pad_len > 0:
            x_seq = torch.cat([x_seq, x_seq[:, -1:].repeat(1, pad_len, 1)], dim=1)
            v_seq = torch.cat([v_seq, v_seq[:, -1:].repeat(1, pad_len, 1)], dim=1)

        new_L = x_seq.shape[1]
        num_blocks = new_L // block_size

        # 1. Aggregación local por bloques
        x_blocks = x_seq.view(B * num_blocks, block_size, D)
        v_blocks = v_seq.view(B * num_blocks, block_size, D)
        x_local, v_local, local_weights = self.local_pool(x_blocks, v_blocks)

        x_local = x_local.view(B, num_blocks, D)
        v_local = v_local.view(B, num_blocks, D)

        # 2. Aggregación global entre bloques
        x_agg, v_agg, _ = self.global_pool(x_local, v_local)

        return x_agg, v_agg, local_weights
