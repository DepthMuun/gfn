"""
Stochastic Forces — GFN V5
Módulo unificado para fuerzas físicas aleatorias (Langevin dynamics).
"""

import torch
import torch.nn as nn
from typing import Optional
from ...interfaces.physics import PhysicsEngine

class BrownianForce(nn.Module):
    """
    Inyecta una fuerza browniana isótropa (ruido blanco).
    Simula el término estocástico en la ecuación de Langevin.
    """
    def __init__(self, sigma: float = 0.01):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Calcula la perturbación estocástica.
        Args:
            x, v: [Batch, (Heads), Dim] tensores de estado
            dt: paso de integración
        Returns:
            Fuerza aleatoria [Batch, (Heads), Dim]
        """
        is_valid_dt = (dt > 0)
        if isinstance(dt, torch.Tensor):
            safe_dt = torch.clamp(dt, min=1e-8)
        else:
            safe_dt = max(float(dt), 1e-8)
        
        # La aceleración estocástica escala con 1/sqrt(dt) para que al integrar (a*dt)
        # el desplazamiento resultante escale con sqrt(dt).
        noise = torch.randn_like(v) * (self.sigma * (safe_dt ** -0.5))
        
        if isinstance(is_valid_dt, torch.Tensor):
             return torch.where(is_valid_dt, noise, torch.zeros_like(v))
        return noise if is_valid_dt else torch.zeros_like(v)

class OUDynamicsForce(nn.Module):
    """
    Ornstein-Uhlenbeck process force.
    Añade reversión a la media al ruido, útil para exploración local suave.
    """
    def __init__(self, sigma: float = 0.01, theta: float = 0.15, mu: float = 0.0):
        super().__init__()
        self.sigma = sigma
        self.theta = theta    # Mean reversion speed
        self.mu = mu          # Mean reversion level
        self._prev_noise: Optional[torch.Tensor] = None

    def reset(self):
        self._prev_noise = None

    def forward(self, x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        if self._prev_noise is None or self._prev_noise.shape != v.shape:
            self._prev_noise = torch.zeros_like(v)
            
        is_valid_dt = (dt > 0)
        if isinstance(dt, torch.Tensor):
            safe_dt = torch.clamp(dt, min=1e-8)
        else:
            safe_dt = max(float(dt), 1e-8)

        noise = torch.randn_like(v)
        # OU step: x_t = x_{t-1} + theta * (mu - x_{t-1}) * dt + sigma * sqrt(dt) * N(0,1)
        # Adaptado aquí como una "fuerza" temporal que se suma a la aceleración
        next_noise = (
            self._prev_noise 
            + self.theta * (self.mu - self._prev_noise) * safe_dt 
            + self.sigma * (safe_dt ** -0.5) * noise
        )
        
        if isinstance(is_valid_dt, torch.Tensor):
            self._prev_noise = torch.where(is_valid_dt, next_noise, self._prev_noise)
            return torch.where(is_valid_dt, self._prev_noise, torch.zeros_like(v))
            
        if is_valid_dt:
            self._prev_noise = next_noise
            return self._prev_noise
        return torch.zeros_like(v)
