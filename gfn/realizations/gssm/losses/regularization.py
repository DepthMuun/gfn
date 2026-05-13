"""
Regularization Losses — GFN V5
Módulo para pérdidas auxiliares exógenas como Balance Dinámico y Simetrías de Noether.
Migrado y modernizado desde gfn_old/losses/physics/
"""

import torch
import torch.nn as nn
from typing import List, Optional
from ..constants import EPS

class NoetherSymmetryLoss(nn.Module):
    """
    Semantic Symmetry (Noether) Loss.
    Enforces that computationally symmetric pathways (like isomorphic attention heads)
    preserve equivalent geometric dynamics (Christoffel symbols or accelerations).
    """
    def __init__(self, lambda_n: float = 0.01):
        super().__init__()
        self.lambda_n = lambda_n

    def forward(self, accelerations: torch.Tensor, isomeric_groups: List[List[int]]) -> torch.Tensor:
        """
        Calcula la penalización por asimetría entre flujos paralelos.
        
        Args:
            accelerations: Tensor de aceleraciones [Batch, Heads, HeadDim]
            isomeric_groups: Grupos de índices de cabezas que deben comportarse idénticamente.
                             Ej: [[0, 1], [2, 3]]
        """
        if not isomeric_groups or accelerations.dim() != 3:
            return torch.zeros(1, device=accelerations.device, requires_grad=True)
            
        total_diff = torch.zeros(1, device=accelerations.device, requires_grad=True)
        count = 0
        
        for group in isomeric_groups:
            if len(group) < 2: 
                continue
            
            # Use the mean of the group as the geometric reference (more stable than star topology)
            group_accels = accelerations[:, group, :]  # [Batch, GroupSize, HeadDim]
            mean_accel = group_accels.mean(dim=1, keepdim=True)  # [Batch, 1, HeadDim]
            
            # Penalize deviation from the group mean
            group_diff = (group_accels - mean_accel).pow(2).mean()
            total_diff = total_diff + group_diff
            count += 1
                
        if count == 0:
            return torch.zeros(1, device=accelerations.device, requires_grad=True)
            
        return self.lambda_n * (total_diff / float(count))


class DynamicLossBalancer(nn.Module):
    """
    Dynamically balances loss components based on batch statistics.
    Prevents a single physical/generative loss term from dominating the gradient manifold.
    """
    def __init__(self, target_ratio: float = 1.0):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, loss_components: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Pondera las pérdidas dinámicamente.
        Args:
            loss_components: Lista de tensores escalares de pérdida.
        """
        if len(loss_components) <= 1:
            return loss_components
            
        # Calcula escalas basadas en las normas sin afectar el grafo de cómputo original
        with torch.no_grad():
            norms = torch.stack([l.detach().abs() + EPS for l in loss_components])
            mean_norm = norms.mean()
            scales = self.target_ratio * mean_norm / norms
            
        # Multiplica la pérdida original por su escala detached para que fluya el gradiente
        return [l * s for l, s in zip(loss_components, scales)]
