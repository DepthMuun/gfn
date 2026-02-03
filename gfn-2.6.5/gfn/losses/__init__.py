"""
Losses
======

Physics-informed loss functions for stable geodesic training.

Available Losses:
    - hamiltonian_loss: Energy conservation
    - geodesic_regularization: Curvature smoothness
    - kinetic_energy_penalty: Velocity regularization
    - noether_loss: Semantic symmetry enforcement
    - curiosity_loss: Entropy-driven exploration
    - GFNLoss: Combined loss (main)
"""

from .hamiltonian import hamiltonian_loss
from .geodesic import geodesic_regularization
from .kinetic import kinetic_energy_penalty
from .noether import noether_loss
from .curiosity import curiosity_loss
from .combined import GFNLoss

# Topology-specific losses
from .circular import circular_distance_loss, CircularDistanceLoss
from .toroidal import toroidal_distance_loss, ToroidalDistanceLoss

__all__ = [
    'hamiltonian_loss',
    'geodesic_regularization',
    'kinetic_energy_penalty',
    'noether_loss',
    'curiosity_loss',
    'GFNLoss',
    'circular_distance_loss',
    'CircularDistanceLoss',
    'toroidal_distance_loss',
    'ToroidalDistanceLoss',
]
