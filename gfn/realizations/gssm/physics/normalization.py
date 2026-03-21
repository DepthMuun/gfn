"""
gfn/physics/normalization.py — GFN V5
Portado desde: gfn_old/nn/layers/physics/normalization.py

Registry centralizado de normalizaciones dependientes de la geometría del manifold.
Principio: las normalizaciones de POSICIÓN dependen de la topología; las de VELOCIDAD
siempre son Euclidianas (espacio tangente).
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from ..constants import MAX_VELOCITY, EPSILON_STANDARD, TOPOLOGY_TORUS


class BaseManifoldNormalization(nn.Module, ABC):
    """Abstract base class for geometry-aware normalization layers."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TorusPositionNormalization(BaseManifoldNormalization):
    """
    Envuelve la posición de forma isométrica en [-π, π].
    Preserva la topología toroidal: atan2(sin(x), cos(x)).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))


class TangentVelocityNormalization(nn.Module):
    """
    RMSNorm con clamp de velocidad máxima para el espacio tangente.
    Previene la aceleración descontrolada en regiones de alta curvatura.
    """
    def __init__(self, dim: int, eps: float = EPSILON_STANDARD):
        super().__init__()
        self.rms = nn.RMSNorm(dim, eps=eps)
        self.max_v = MAX_VELOCITY

    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Nota: x es la velocidad a normalizar. context_x es la posición (opcional para metric-aware)
        x = torch.clamp(x, -self.max_v, self.max_v)
        return self.rms(x)


class MetricAwareVelocityNormalization(nn.Module):
    """
    Normalización que escala la velocidad basándose en la métrica Riemanniana.
    Asegura que la norma geodésica ||v||_g no exceda el límite físico.
    """
    def __init__(self, dim: int, geometry=None, max_v: float = MAX_VELOCITY):
        super().__init__()
        self.geometry = geometry
        self.max_v = max_v
        self.rms = nn.RMSNorm(dim)  # Fallback si no hay geometría

    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.geometry is not None and context_x is not None:
            # x: [B, D] (velocidad), context_x: [B, D] (posición)
            # 1. Obtener tensor métrico g(context_x)
            g = self.geometry.metric_tensor(context_x)  # [B, D, D] or [B, D] or [D, D]
            
            # 2. Calcular norma cuadrada: x^T g x
            # Usamos broadcast matmul para mayor robustez que bmm
            # v: [B, D] -> [B, 1, D]
            v_exp = x.unsqueeze(1)
            if g.dim() == 2 and g.shape[0] == x.shape[0]: # [B, D] - métrica diagonal
                norm_sq = (x * g * x).sum(dim=-1, keepdim=True).unsqueeze(-1)
            elif g.dim() == 2: # [D, D] - métrica constante
                g_exp = g.unsqueeze(0) # [1, D, D]
                norm_sq = v_exp @ g_exp @ v_exp.transpose(1, 2)
            else: # [B, D, D]
                norm_sq = v_exp @ g @ v_exp.transpose(1, 2)
            
            norm_g = torch.sqrt(norm_sq.squeeze(-1).squeeze(-1) + 1e-8)
            
            # 3. Escalar si excede max_v
            scale = torch.clamp(self.max_v / norm_g, max=1.0)
            x = x * scale.unsqueeze(-1)
            
            # 4. Retornar velocidad clampada pero con magnitud preservada
            return x
            
        # Fallback a clamp estándar
        return torch.clamp(x, -self.max_v, self.max_v)


class EuclideanPositionNormalization(BaseManifoldNormalization):
    """
    Identidad para posiciones Euclidianas.
    Físicamente más seguro que RMSNorm para coordenadas de posición.
    """
    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class IdentityNormalization(BaseManifoldNormalization):
    """Pass-through — usar cuando ninguna normalización es requerida."""
    def forward(self, x: torch.Tensor, context_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class ManifoldNormalizationRegistry:
    """
    Registry centralizado para obtener la normalización apropiada
    según el tipo de variable física y la topología del manifold.

    Tipos disponibles:
      'position_torus'    — atan2(sin, cos) wrapping
      'position_euclidean'— Identidad (seguro para coordenadas Euclidianas)
      'velocity_tangent'  — RMSNorm clampado (espacio tangente)
      'velocity_metric'   — MetricAware norm (estricto)
      'feature_hidden'    — Igual a velocity_tangent (features internas)
      'identity'          — Sin transformación
    """
    _REGISTRY = {
        'position_torus':    TorusPositionNormalization,
        'position_euclidean': EuclideanPositionNormalization,
        'velocity_tangent':  TangentVelocityNormalization,
        'velocity_metric':   MetricAwareVelocityNormalization,
        'feature_hidden':    TangentVelocityNormalization,
        'identity':          IdentityNormalization,
    }

    @classmethod
    def get(cls, norm_type: str, dim: int = 64, geometry=None) -> nn.Module:
        norm_cls = cls._REGISTRY.get(norm_type.lower(), IdentityNormalization)
        if norm_cls in (TangentVelocityNormalization,):
            return norm_cls(dim)
        if norm_cls == MetricAwareVelocityNormalization:
            return norm_cls(dim, geometry=geometry)
        return norm_cls()

    @classmethod
    def get_for_topology(cls, topology: str, dim: int = 64, 
                         is_velocity: bool = False, geometry=None) -> nn.Module:
        """
        Atajo: selecciona automáticamente la normalización correcta
        basándose en la topología y si es posición o velocidad.
        """
        if is_velocity:
            # Si hay geometría disponible, usamos la estricta
            if geometry is not None:
                return cls.get('velocity_metric', dim, geometry=geometry)
            return cls.get('velocity_tangent', dim)
        if topology.lower().strip() == TOPOLOGY_TORUS:
            return cls.get('position_torus', dim)
        return cls.get('position_euclidean', dim)


__all__ = [
    'ManifoldNormalizationRegistry',
    'TorusPositionNormalization',
    'TangentVelocityNormalization',
    'EuclideanPositionNormalization',
    'IdentityNormalization',
]
