import torch
import torch.nn as nn
from typing import Optional
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from .base import BaseDynamics


class MixDynamics(BaseDynamics):
    """
    Mix Dynamics: state_next = norm(alpha * current + (1 - alpha) * proposal).
    
    alpha es un parámetro aprendible que controla la memoria del estado anterior.
    Usamos log_alpha para evitar saturación y permitir exploración del espacio de interpolación.
    
    Inicialización recomendada: alpha cercano a 0.0 para dar más peso a la propuesta inicial,
    luego el modelo aprende gradualmente el balance óptimo.
    """
    def __init__(self, dim: int, norm_layer=None, topology: str = TOPOLOGY_EUCLIDEAN,
                 alpha_init: float = 0.3, **kwargs):
        super().__init__(dim, norm_layer, topology, **kwargs)
        # Usar log_alpha para evitar saturación de sigmoid
        # alpha = sigmoid(log_alpha) → rango completo (0, 1)
        self.log_alpha = nn.Parameter(torch.tensor([alpha_init]))
        
        # Escala del cambio para estabilidad
        self.change_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, current_state: torch.Tensor,
                absolute_proposal: torch.Tensor, 
                context_x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Convertir log_alpha a alpha en (0, 1)
        alpha = torch.sigmoid(self.log_alpha)
        
        # Interpolación entre estado actual y propuesta
        if self.topology == TOPOLOGY_TORUS:
            # Interpolación Geodésica (Circular Slerp)
            # Promediamos en el espacio de embedding (sin, cos) y volvemos a ángulo
            interpolated = torch.atan2(
                alpha * torch.sin(current_state) + (1.0 - alpha) * torch.sin(absolute_proposal),
                alpha * torch.cos(current_state) + (1.0 - alpha) * torch.cos(absolute_proposal)
            )
        else:
            # Interpolación Euclidiana estándar
            interpolated = alpha * current_state + (1.0 - alpha) * absolute_proposal
        
        # Aplicar normalización según topología (context_x permite metric-aware)
        result = self._apply_norm(interpolated, context_x=context_x)
        
        # Aplicar escala del cambio (learning rate suave)
        if self.topology == TOPOLOGY_TORUS:
            # En el toro, la diferencia también es circular
            diff = torch.atan2(torch.sin(result - current_state), torch.cos(result - current_state))
            result = current_state + self.change_scale * diff
        else:
            result = current_state + self.change_scale * (result - current_state)
        
        return result
    
    def get_alpha(self) -> float:
        """Retorna el valor actual de alpha para debugging."""
        return float(torch.sigmoid(self.log_alpha).item())
