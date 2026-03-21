"""
GFN Realizations API Router (Purified Version)
==============================================
Agnostic factory and dynamic registry for GFN architectural realizations.
Follows SOLID principles: open for extension, closed for modification.
"""

import logging
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import torch.nn as nn

logger = logging.getLogger(__name__)

@runtime_checkable
class RealizationProvider(Protocol):
    """Protocol defining the interface any GFN realization must provide."""
    def create(self, **kwargs) -> nn.Module: ...
    def save(self, model: nn.Module, path: str): ...
    def load(self, path: str, **kwargs) -> nn.Module: ...

# The Dynamic Registry
_REGISTRY: Dict[str, RealizationProvider] = {}

def register(name: str, provider: RealizationProvider):
    """
    Register a new realization architecture.
    
    Args:
        name: Unique identifier for the realization.
        provider: An object or module implementing the RealizationProvider protocol.
    """
    name = name.lower()
    if name in _REGISTRY:
        logger.debug(f"Overwriting GFN realization provider: {name}")
    _REGISTRY[name] = provider

def list_available() -> List[str]:
    """List all dynamically registered architectural realizations."""
    return list(_REGISTRY.keys())

def create(name: str, **kwargs) -> nn.Module:
    """
    Unified factory to create any registered GFN realization by name.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"GFN Error: Realization '{name}' is not registered. "
            f"Ensure the subpackage is imported. Available: {list_available()}"
        )
    return _REGISTRY[name].create(**kwargs)

def save(model: nn.Module, path: str, realization: Optional[str] = None):
    """Unified save interface delegate."""
    if realization and realization.lower() in _REGISTRY:
        _REGISTRY[realization.lower()].save(model, path)
    else:
        import torch
        torch.save(model.state_dict(), path)

def load(path: str, realization: str, **kwargs) -> nn.Module:
    """Unified load interface delegate."""
    realization = realization.lower()
    if realization not in _REGISTRY:
        raise ValueError(f"GFN Error: Realization provider for '{realization}' not found.")
    return _REGISTRY[realization].load(path, **kwargs)
