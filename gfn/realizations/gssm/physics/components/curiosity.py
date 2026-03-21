"""
Curiosity Force (Intrinsic Motivation) — GFN V5
Inyecta fuerzas dirigidas lejos de estados de alta densidad (exploración en el Manifold).
"""

import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN

class GeometricCuriosityForce(nn.Module):
    """
    Aplica una fuerza repulsiva de los densos agregados geométricos, 
    usando la curvatura local o historial de posiciones.
    En V5, actúa como un plugin acoplado al PhysicsEngine.
    """
    def __init__(self, strength: float = 0.1, decay: float = 0.99):
        super().__init__()
        self.strength = strength
        self.decay = decay
        # Para exploración simplificada, repelimos del centro de masa del batch
        # Opciones más avanzadas requerirían un Density Estimator externo.
        
    def forward(self, x: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calcula la fuerza repulsiva de curiosidad basada en la dispersión actual del batch.
        """
        if self.strength <= 0.0:
            return torch.zeros_like(v)
        
        # 1. Encontrar el 'atractor' trivial del batch (Centro de gravedad)
        if kwargs.get('topology', TOPOLOGY_EUCLIDEAN) == TOPOLOGY_TORUS:
            # Mean Circular: atan2(mean(sin), mean(cos))
            sin_x = torch.sin(x); cos_x = torch.cos(x)
            batch_center = torch.atan2(sin_x.mean(dim=0, keepdim=True), cos_x.mean(dim=0, keepdim=True))
            # 2. Vector de escape (dirección geodésica)
            direction = x - batch_center
            direction = torch.atan2(torch.sin(direction), torch.cos(direction))
        else:
            batch_center = x.mean(dim=0, keepdim=True)
            # 2. Vector de escape (dirección euclídea)
            direction = x - batch_center
        
        # 3. Fuerza inversamente proporcional a la distancia
        dist_sq = (direction ** 2).sum(dim=-1, keepdim=True) + 1e-6
        repulsion_mag = self.strength / dist_sq
        
        # 4. Normalizar dirección y escalar
        force = (direction / (dist_sq ** 0.5 + 1e-8)) * repulsion_mag
        
        # Limitar la fuerza máxima para evitar inestabilidad
        return torch.clamp(force, min=-5.0, max=5.0)
