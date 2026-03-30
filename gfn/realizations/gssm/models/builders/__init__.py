"""
gfn/realizations/gssm/models/builders/__init__.py
Component Builder pattern for ModelFactory.

Desmonolitiza ModelFactory en builders especializados, cada uno
responsable de crear un componente específico del modelo.
"""
from typing import Any, Dict, Type, Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ComponentBuilder(ABC):
    """
    Base class for all model component builders.
    
    Each builder is responsible for creating a specific component
    of the ManifoldModel from a ManifoldConfig.
    """
    
    def __init__(self, config: Any):
        self.config = config
    
    @abstractmethod
    def build(self) -> nn.Module:
        """Build and return the component."""
        pass
    
    def validate_config(self) -> None:
        """Override to validate configuration before building."""
        pass


class BuilderRegistry:
    """Registry for component builders."""
    
    def __init__(self):
        self._builders: Dict[str, Type[ComponentBuilder]] = {}
    
    def register(self, key: str, builder_cls: Type[ComponentBuilder]) -> None:
        """Register a builder class."""
        if key in self._builders:
            raise ValueError(f"Builder '{key}' already registered")
        self._builders[key] = builder_cls
    
    def get(self, key: str) -> Type[ComponentBuilder]:
        """Get builder class by key."""
        if key not in self._builders:
            available = list(self._builders.keys())
            raise KeyError(f"Builder '{key}' not found. Available: {available}")
        return self._builders[key]
    
    def list_builders(self) -> list[str]:
        """List all registered builder keys."""
        return list(self._builders.keys())


# Global builder registry
MODEL_BUILDER_REGISTRY = BuilderRegistry()


# Import builders to register them
from .embedding_builder import EmbeddingBuilder
from .layer_builder import LayerBuilder
from .readout_builder import ReadoutBuilder
from .plugin_builders import PoolingBuilder, CheckpointingBuilder, AdjointBuilder, LensingBuilder


__all__ = [
    'ComponentBuilder',
    'BuilderRegistry',
    'MODEL_BUILDER_REGISTRY',
    'EmbeddingBuilder',
    'LayerBuilder',
    'ReadoutBuilder',
    'PoolingBuilder',
    'CheckpointingBuilder',
    'AdjointBuilder',
    'LensingBuilder',
]
