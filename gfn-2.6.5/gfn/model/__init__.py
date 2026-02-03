"""
Manifold Model Package
=====================

Modular architecture for the Manifold sequence model.

Structure:
    - state.py: ManifoldState container and utilities
    - fusion.py: CUDAFusionManager for kernel fusion

Usage:
    from gfn.model import ManifoldState, CUDAFusionManager
    
Note:
    To import Manifold class, use:
        from gfn import Manifold
    or:
        from gfn.model_core import Manifold
"""

from .state import ManifoldState
from .fusion import CUDAFusionManager

__all__ = [
    'ManifoldState',
    'CUDAFusionManager',
]
