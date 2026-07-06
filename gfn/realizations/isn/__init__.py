"""
Public package entrypoint for the Inertial State Network realization.
"""

# 1. Initialize registries and interfaces
from . import registry
from . import interfaces

# 2. Import and register all components
from . import physics
from . import projections
from . import training

# 3. Expose public API
from .api import create, save, load
from .models.model import Model

# Register with the central realization registry when available.
try:
    from gfn import api as central_api
    from . import api as isn_api
    central_api.register('isn', isn_api)
except ImportError:
    pass # Standalone usage

__all__ = ["create", "save", "load", "Model", "registry", "training", "physics", "projections"]
