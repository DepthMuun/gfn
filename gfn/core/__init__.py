"""
Core Models
===========

Main model implementations for GFN.

Available Models:
    - Manifold: Main GFN model with Riemannian geometry
    - AdjointManifold: Adjoint variant for memory-efficient training
"""

from .manifold import Manifold
from .adjoint import AdjointManifold

__all__ = [
    'Manifold',
    'AdjointManifold',
]
