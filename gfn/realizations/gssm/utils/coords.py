"""
gfn/utils/coords.py
===================
Coordinate conversions for toroidal manifolds.

The torus uses angles in [-π, π]. Prediction spaces for tasks like
detection use normalized coordinates in [0, 1]. These functions convert
between both spaces in a differentiable manner.

Usage:
    from ..utils.coords import box_to_torus, torus_to_box, wrap_angles

    # In training:
    target_angles = box_to_torus(labels_01)     # [0,1] → [-π, π]
    criterion(pred_angles, target_angles)

    # In inference:
    boxes_01 = torus_to_box(pred_angles)         # [-π, π] → [0,1]
"""

import math
import torch

__all__ = ['box_to_torus', 'torus_to_box', 'wrap_angles', 'angle_to_unit']


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    Wraps arbitrary angles to the range [-π, π] in a differentiable way.

    Uses atan2(sin, cos) instead of modulo, allowing the gradient
to flow correctly through the operation.

    Args:
        angles: Tensor of any shape, values in any range.

    Returns:
        Tensor of the same shape, values in [-π, π].
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def box_to_torus(coords_01: torch.Tensor) -> torch.Tensor:
    """
    Converts normalized coordinates [0, 1] to toroidal angles [-π, π].

    Linear mapping: 0 → -π,  0.5 → 0,  1 → π

    Args:
        coords_01: Tensor [..., N] with values in [0, 1].

    Returns:
        Tensor [..., N] with values in [-π, π].
    """
    return coords_01.clamp(0.0, 1.0) * (2.0 * math.pi) - math.pi


def torus_to_box(angles: torch.Tensor) -> torch.Tensor:
    """
    Converts toroidal angles [-π, π] to normalized coordinates [0, 1].

    Applies wrap_angles first to handle out-of-range angles,
    then maps [-π, π] → [0, 1].

    Args:
        angles: Tensor [..., N] with values nominally in [-π, π]
                (overflow is handled with wrap).

    Returns:
        Tensor [..., N] with values in [0, 1].
    """
    wrapped = wrap_angles(angles)
    return (wrapped + math.pi) / (2.0 * math.pi)


def angle_to_unit(angle: torch.Tensor) -> torch.Tensor:
    """
    Converts a scalar angle to a confidence representation in [0, 1].

    Used to convert the objectness angle of a toroidal manifold
    to a detection probability:
        θ = 0   → confidence = 0.0  (no drone)
        θ = π/2 → confidence = 0.5
        θ = π   → confidence = 1.0  (drone with certainty)

    Formula: conf = (-cos(θ) + 1) / 2

    Args:
        angle: Scalar tensor or [B] — objectness angle.

    Returns:
        Tensor of the same shape — probability in [0, 1].
    """
    return (-torch.cos(angle) + 1.0) / 2.0
