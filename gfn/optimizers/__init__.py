"""
Riemannian Optimizers
=====================

Optimizers designed for training on curved manifolds.

Available Optimizers:
    - RiemannianAdam: Adam with Riemannian retraction
    - ManifoldSGD: Simple SGD with retraction

Retraction Types:
    - 'euclidean': Standard Euclidean update (no retraction)
    - 'normalize': Normalize to bounded manifold
    - 'torus': Toroidal manifold (periodic)
    - 'cayley': Cayley retraction (orthogonal-ish)
"""

from .riemannian_adam import RiemannianAdam
from .manifold_sgd import ManifoldSGD

__all__ = [
    'RiemannianAdam',
    'ManifoldSGD',
]
