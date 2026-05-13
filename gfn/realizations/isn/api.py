"""
ISN API — GFN V4/V5
Public interface for the Inertial State Network realization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .models.model import Model
from .components.scanners.linear_scanner import LinearScanner
from .components.worlds.topological_world import TopologicalWorld
from .components.emitters.threshold_emitter import ThresholdEmitter

def create(
    vocab_size: int = 5000,
    d_model: int = 512,
    d_embedding: int = 256,
    d_properties: int = 64,
    scanner_cls = LinearScanner,
    world_cls = TopologicalWorld,
    emitter_cls = ThresholdEmitter,
    scanner_kwargs: Optional[Dict[str, Any]] = None,
    world_kwargs: Optional[Dict[str, Any]] = None,
    emitter_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Model:
    """
    Factory to create an ISN model with modular components and specific configurations.
    """
    s_kwargs = scanner_kwargs or {}
    w_kwargs = world_kwargs or {}
    e_kwargs = emitter_kwargs or {}
    
    scanner = scanner_cls(vocab_size=vocab_size, d_model=d_model, **s_kwargs)
    world = world_cls(d_model=d_model, d_embedding=d_embedding, d_properties=d_properties, **w_kwargs)
    emitter = emitter_cls(d_embedding=d_embedding, vocab_size=vocab_size, **e_kwargs)
    
    return Model(
        scanner=scanner,
        world=world,
        emitter=emitter,
        hooks=kwargs.get('hooks')
    )

def save(model: nn.Module, path: str):
    """Save model weights."""
    torch.save(model.state_dict(), path)

def load(path: str, vocab_size: int, **kwargs) -> Model:
    """Load model from weights."""
    model = create(vocab_size=vocab_size, **kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
