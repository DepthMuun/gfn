import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class ResidualDynamics(BaseDynamics):
    """
    Residual (Skip-Connection) Dynamics.
    
    Implementa una conexión residual real:
    state_next = current_state + scale * norm(proposal - current_state)
    
    Esto permite que el modelo aprenda una corrección sobre el estado actual
    en lugar de reemplazarlo completamente (como direct) o interpolarlo (como mix).
    
    Para POSICIÓN (torus): wrapping a [-π, π] después del residual.
    Para VELOCIDAD (euclidean): RMSNorm sobre el residual.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN, 
                 residual_scale: float = 0.1, **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        # Escala del residual - parámetro aprendible pero inicialmente pequeño
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Calcular residual: diferencia entre propuesta y estado actual
        if self.topology == TOPOLOGY_TORUS:
            # Diferencia geodésica en el toro
            residual = torch.atan2(torch.sin(absolute_proposal - current_state),
                                   torch.cos(absolute_proposal - current_state))
        else:
            residual = absolute_proposal - current_state
        
        # Aplicar normalización al residual
        # Nota: para velocities, el context_x (posición) permite MetricNormalization
        residual_normalized = self._apply_norm(residual, context_x=context_x)
        
        # Escalar el residual con parámetro aprendible
        scale = torch.sigmoid(self.residual_scale)
        
        # Aplicar conexión residual: state + scale * residual
        next_state = current_state + scale * residual_normalized
        
        # En el toro, asegurar que seguimos en [-π, π]
        if self.topology == TOPOLOGY_TORUS:
            next_state = torch.atan2(torch.sin(next_state), torch.cos(next_state))
            
        return next_state
