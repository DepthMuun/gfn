"""
ISN — Inertial State Network (GFN V4)
Modular, scalable, physics-based simulator.
"""

import os
import sys
from . import api as isn_api
from .. import api as central_api

# Register with central realization registry
central_api.register('isn', isn_api)

# 1. NATIVE KERNEL BOOTSTRAPPING
# Locating and injecting the compiled C++ topology engine into sys.path
_CSRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'csrc/topology'))
if _CSRC_PATH not in sys.path:
    sys.path.append(_CSRC_PATH)

try:
    import gfn_topology
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    # (Optional) Log warning: print("GFN WARNING: Native core missing. Performance throttled.")

# 2. PUBLIC API
from .api import create, save, load
from .models.model import Model
from .interfaces import BaseScanner, BaseWorldEngine, BaseEmitter
from .hooks import ISNHook, HookManager
from .training.trainer import Trainer

# Components
from .components.scanners.linear_scanner import LinearScanner
from .components.scanners.transformer_scanner import TransformerScanner
from .components.scanners.ssm_scanner import SSMScanner
from .components.scanners.gfn_scanner import GFNScanner
from .components.worlds.topological_world import TopologicalWorld
from .components.worlds.gfn_world import GFNWorld
from .components.emitters.threshold_emitter import ThresholdEmitter
from .components.emitters.ssm_emitter import SSMEmitter
from .components.emitters.gfn_emitter import GFNEmitter

__all__ = [
    'create', 
    'save', 
    'load', 
    'Model', 
    'BaseScanner', 
    'BaseWorldEngine', 
    'BaseEmitter',
    'ISNHook',
    'HookManager',
    'Trainer',
    'LinearScanner',
    'TransformerScanner',
    'SSMScanner',
    'GFNScanner',
    'TopologicalWorld',
    'GFNWorld',
    'ThresholdEmitter',
    'SSMEmitter',
    'GFNEmitter',
]
