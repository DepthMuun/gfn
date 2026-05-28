"""
gfn/realizations/gssm/models/plugins/__init__.py
Plugin system for ManifoldLayer components.
"""
from typing import Dict, Type, Any, Optional, Callable, TypeVar, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

T = TypeVar("T")


class LayerPlugin(ABC, nn.Module):
    """
    Base class for ManifoldLayer plugins.
    
    Each plugin can hook into:
    - __init__: Initialize plugin-specific parameters
    - pre_integrate: Modify x, v, dt before integrator step
    - post_integrate: Modify x, v after integrator step
    - pre_mix: Modify x, v before mixing
    - post_mix: Modify x, v after mixing
    - finalize: Final modifications before return
    """
    
    def __init__(self, layer: nn.Module, config: Any):
        super().__init__()
        # Store the layer without registering it as a sub‑module to avoid recursive `.to()` loops.
        object.__setattr__(self, "_layer", layer)
        self.config = config
        self.enabled = True

    @property
    def layer(self) -> nn.Module:
        """Return the associated ManifoldLayer without registering it as a child module."""
        return getattr(self, "_layer")
    
    def setup(self) -> None:
        """Called after plugin is attached to layer. Override to initialize parameters."""
        pass
    
    def pre_integrate(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        dt: torch.Tensor,
        force: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hook before integrator step.
        Returns: (x_modified, v_modified, dt_modified)
        """
        return x, v, dt
    
    def post_integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        x_prev: torch.Tensor,
        v_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hook after integrator step.
        Returns: (x_modified, v_modified)
        """
        return x, v
    
    def pre_mix(
        self,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hook before head mixing."""
        return x, v
    
    def post_mix(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        x_mix: torch.Tensor,
        v_mix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hook after head mixing.
        Returns: (x_modified, v_modified)
        """
        return x_mix, v_mix
    
    def finalize(
        self,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Final hook before returning from forward."""
        return x, v


class LayerPluginRegistry:
    """Registry for layer plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Type[LayerPlugin]] = {}
    
    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a plugin class."""
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._plugins:
                raise ValueError(f"Plugin '{key}' already registered")
            if not issubclass(cls, LayerPlugin):
                raise TypeError(f"Plugin must inherit from LayerPlugin, got {cls}")
            self._plugins[key] = cls
            return cls
        return decorator
    
    def get(self, key: str) -> Type[LayerPlugin]:
        """Get plugin class by key."""
        if key not in self._plugins:
            available = list(self._plugins.keys())
            raise KeyError(f"Plugin '{key}' not found. Available: {available}")
        return self._plugins[key]
    
    def list_plugins(self) -> list[str]:
        """List all registered plugin keys."""
        return list(self._plugins.keys())
    
    def create_plugin(
        self,
        key: str,
        layer: nn.Module,
        config: Any
    ) -> Optional[LayerPlugin]:
        """Create plugin instance if enabled in config."""
        plugin_cls = self.get(key)
        
        # Check if plugin is enabled in config
        plugin_config = getattr(config, key, None)
        if plugin_config is None:
            # Try nested config
            parts = key.split('_')
            current = config
            for part in parts:
                current = getattr(current, part, None)
                if current is None:
                    break
            plugin_config = current
        
        # Check enabled flag
        is_enabled = True
        if hasattr(plugin_config, 'enabled'):
            is_enabled = plugin_config.enabled
        elif isinstance(plugin_config, dict):
            is_enabled = plugin_config.get('enabled', True)
        elif plugin_config is None:
            is_enabled = False
        
        if not is_enabled:
            return None
        
        plugin = plugin_cls(layer, plugin_config if plugin_config else config)
        plugin.enabled = True
        return plugin


# Global layer plugin registry
LAYER_PLUGIN_REGISTRY = LayerPluginRegistry()


def register_layer_plugin(key: str):
    """Decorator to register a layer plugin."""
    return LAYER_PLUGIN_REGISTRY.register(key)


# Export
__all__ = [
    'LayerPlugin',
    'LayerPluginRegistry',
    'LAYER_PLUGIN_REGISTRY',
    'register_layer_plugin'
]
