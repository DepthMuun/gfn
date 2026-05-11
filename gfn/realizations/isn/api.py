"""
ISN API — Modular V5
===================
Public interface for the Inertial State Network realization.
Uses the Registry system for dynamic component assembly.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .models.model import Model
from .registry import physics, scanners, emitters

def create(
    vocab_size: int = 5000,
    d_model: int = 512,
    d_embedding: int = 256,
    d_properties: int = 64,
    scanner: str = "gfn",
    world: str = "gfn",
    emitter: str = "gfn",
    scanner_kwargs: Optional[Dict[str, Any]] = None,
    world_kwargs: Optional[Dict[str, Any]] = None,
    emitter_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Model:
    """
    Factory to create an ISN model using registered components.
    """
    s_kwargs = scanner_kwargs or {}
    w_kwargs = world_kwargs or {}
    e_kwargs = emitter_kwargs or {}
    
    # Fetch classes from registry
    scanner_cls = scanners.get(scanner)
    world_cls = physics.get(world)
    emitter_cls = emitters.get(emitter)
    
    # Instantiate components with ALL dimensions for robustness
    scanner_inst = scanner_cls(
        vocab_size=vocab_size, 
        d_model=d_model, 
        d_embedding=d_embedding, 
        d_properties=d_properties, 
        **s_kwargs
    )
    world_inst = world_cls(
        d_model=d_model, 
        d_embedding=d_embedding, 
        d_properties=d_properties, 
        **w_kwargs
    )
    emitter_inst = emitter_cls(
        d_embedding=d_embedding, 
        vocab_size=vocab_size, 
        d_model=d_model,
        d_properties=d_properties,
        **e_kwargs
    )
    
    return Model(
        scanner=scanner_inst,
        world=world_inst,
        emitter=emitter_inst,
        hooks=kwargs.get('hooks')
    )

def save(model: nn.Module, path: str):
    """Save model weights."""
    torch.save(model.state_dict(), path)

def load(path: str, vocab_size: int, **kwargs) -> Model:
    """Load model from weights."""
    model = create(vocab_size=vocab_size, **kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model
