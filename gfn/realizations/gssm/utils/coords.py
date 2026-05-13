"""
gfn/utils/coords.py
===================
Conversiones de coordenadas para manifolds toroidales.

El toro usa ángulos en [-π, π]. El espacio de predicción de tasks como
detección usa coordenadas normalizadas en [0, 1]. Estas funciones convierten
entre ambos espacios de forma diferenciable.

Uso:
    from ..utils.coords import box_to_torus, torus_to_box, wrap_angles

    # En training:
    target_angles = box_to_torus(labels_01)     # [0,1] → [-π, π]
    criterion(pred_angles, target_angles)

    # En inferencia:
    boxes_01 = torus_to_box(pred_angles)         # [-π, π] → [0,1]
"""

import math
import torch

__all__ = ['box_to_torus', 'torus_to_box', 'wrap_angles', 'angle_to_unit']


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    Envuelve ángulos arbitrarios al rango [-π, π] de forma diferenciable.

    Usa atan2(sin, cos) en lugar de módulo, lo que hace que el gradiente
    fluya correctamente a través de la operación.

    Args:
        angles: Tensor de cualquier forma, valores en cualquier rango.

    Returns:
        Tensor de la misma forma, valores en [-π, π].
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def box_to_torus(coords_01: torch.Tensor) -> torch.Tensor:
    """
    Convierte coordenadas normalizadas [0, 1] a ángulos toroidales [-π, π].

    Mapeo lineal: 0 → -π,  0.5 → 0,  1 → π

    Args:
        coords_01: Tensor [..., N] con valores en [0, 1].

    Returns:
        Tensor [..., N] con valores en [-π, π].
    """
    return coords_01.clamp(0.0, 1.0) * (2.0 * math.pi) - math.pi


def torus_to_box(angles: torch.Tensor) -> torch.Tensor:
    """
    Convierte ángulos toroidales [-π, π] a coordenadas normalizadas [0, 1].

    Aplica wrap_angles primero para manejar ángulos fuera de rango,
    luego mapea [-π, π] → [0, 1].

    Args:
        angles: Tensor [..., N] con valores nominalmente en [-π, π]
                (se maneja overflow con wrap).

    Returns:
        Tensor [..., N] con valores en [0, 1].
    """
    wrapped = wrap_angles(angles)
    return (wrapped + math.pi) / (2.0 * math.pi)


def angle_to_unit(angle: torch.Tensor) -> torch.Tensor:
    """
    Convierte un ángulo escalar a una representación de confianza en [0, 1].

    Usado para convertir el ángulo de objectness de un manifold toroidal
    a una probabilidad de detección:
        θ = 0   → confianza = 0.0  (no hay dron)
        θ = π/2 → confianza = 0.5
        θ = π   → confianza = 1.0  (dron con certeza)

    Formula: conf = (-cos(θ) + 1) / 2

    Args:
        angle: Tensor escalar o [B] — ángulo de objectness.

    Returns:
        Tensor de la misma forma — probabilidad en [0, 1].
    """
    return (-torch.cos(angle) + 1.0) / 2.0
