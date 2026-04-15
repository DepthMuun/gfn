"""
ISN Registry System — Modular V5
==============================
Centralized registry for Physics, Projections, and Training Strategies.
"""

from typing import Dict, Type, Any

class Registry:
    """A generic registry for ISN components."""
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Type] = {}

    def register(self, key: str):
        """Decorator to register a class."""
        def wrapper(cls):
            self._entries[key.lower()] = cls
            return cls
        return wrapper

    def get(self, key: str) -> Type:
        """Fetch a class by key."""
        if key.lower() not in self._entries:
            available = ", ".join(self._entries.keys())
            raise ValueError(f"'{key}' not found in {self.name} registry. Available: {available}")
        return self._entries[key.lower()]

    def summary(self) -> str:
        """Return a string summary of registered components."""
        return f"{self.name}: {list(self._entries.keys())}"

# Global Registries
physics = Registry("Physics")
scanners = Registry("Scanners")
emitters = Registry("Emitters")
strategies = Registry("Strategies")

def summary():
    """Returns total registry summary."""
    return "\n".join([
        physics.summary(),
        scanners.summary(),
        emitters.summary(),
        strategies.summary()
    ])
