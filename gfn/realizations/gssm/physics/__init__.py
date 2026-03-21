"""
gfn/physics/__init__.py
Public API for the physics module — GFN V5
"""

from .engine import ManifoldPhysicsEngine
from .monitor import PhysicsMonitorPlugin
from .components import (
    SingularityGate,
    SingularityDetector,
    SingularityRegistry,
    HysteresisModule,
    HysteresisState,
    HysteresisRegistry,
)

__all__ = [
    "ManifoldPhysicsEngine",
    "PhysicsMonitorPlugin",
    # Singularity components
    "SingularityGate",
    "SingularityDetector", 
    "SingularityRegistry",
    # Hysteresis components
    "HysteresisModule",
    "HysteresisState",
    "HysteresisRegistry",
]
