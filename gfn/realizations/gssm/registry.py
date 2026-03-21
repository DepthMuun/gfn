from typing import Dict, Type, Any, Optional, Callable, TypeVar

T = TypeVar("T")

class Registry:
    """A central registry for GFN components."""
    
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Type] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under a given key."""
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._entries:
                raise ValueError(f"Key '{key}' is already registered in {self.name}")
            self._entries[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> Type:
        """Retrieve a registered class by key."""
        if key not in self._entries:
            raise KeyError(f"Key '{key}' not found in {self.name} registry. Available keys: {list(self._entries.keys())}")
        return self._entries[key]

    def list_keys(self) -> list[str]:
        """List all registered keys."""
        return list(self._entries.keys())

# Core Registries
GEOMETRY_REGISTRY = Registry("Geometry")
INTEGRATOR_REGISTRY = Registry("Integrator")
MODEL_REGISTRY = Registry("Model")
LOSS_REGISTRY = Registry("Loss")
COMPONENTS_REGISTRY = Registry("Component")

def register_geometry(key: str):
    return GEOMETRY_REGISTRY.register(key)

def register_integrator(key: str):
    return INTEGRATOR_REGISTRY.register(key)

def register_model(key: str):
    return MODEL_REGISTRY.register(key)

def register_loss(key: str):
    return LOSS_REGISTRY.register(key)

def register_component(key: str):
    return COMPONENTS_REGISTRY.register(key)
