"""
Inertial State Network (ISN) — Modular V5
=======================================
Portable, modular, and extensible implementation of ISN.
"""

# 1. Initialize Registries and Interfaces
from . import registry
from . import interfaces

# 2. Import and Register all Components (Triggers decorators)
from . import physics
from . import projections
from . import training

# 3. Expose Public API
from .api import create, save, load
from .models.model import Model

# Register with central realization registry
try:
    from gfn import api as central_api
    from . import api as isn_api
    central_api.register('isn', isn_api)
except ImportError:
    pass # Standalone usage

__all__ = ["create", "save", "load", "Model", "registry"]
