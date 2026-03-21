"""
gfn/geometry/__init__.py
Public API for the geometry module — GFN V5
"""

# Base and factory
from ..geometry.base import BaseGeometry
from ..geometry.factory import GeometryFactory

# Concrete geometries (imports trigger @register_geometry decorators)
from ..geometry.euclidean import EuclideanGeometry
from ..geometry.torus import ToroidalRiemannianGeometry, FlatToroidalRiemannianGeometry
from ..geometry.low_rank import LowRankRiemannianGeometry, PaperLowRankRiemannianGeometry
from ..geometry.adaptive import AdaptiveRiemannianGeometry
from ..geometry.reactive import ReactiveRiemannianGeometry
from ..geometry.hyperbolic import HyperRiemannianGeometry
from ..geometry.holographic import HolographicRiemannianGeometry
from ..geometry.spherical import SphericalGeometry
from ..geometry.hierarchical import HierarchicalGeometry

# Re-export FrictionGate from unified physics.components location
from ..physics.components.friction import FrictionGate

__all__ = [
    # Base
    "BaseGeometry",
    "GeometryFactory",
    # Implementations
    "EuclideanGeometry",
    "ToroidalRiemannianGeometry",
    "FlatToroidalRiemannianGeometry",
    "LowRankRiemannianGeometry",
    "PaperLowRankRiemannianGeometry",
    "AdaptiveRiemannianGeometry",
    "ReactiveRiemannianGeometry",
    "HyperRiemannianGeometry",
    "HolographicRiemannianGeometry",
    "SphericalGeometry",
    "HierarchicalGeometry",
    # Shared components
    "FrictionGate",
]
