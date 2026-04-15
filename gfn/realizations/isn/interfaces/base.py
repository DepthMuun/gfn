"""
ISN Base Interfaces — Modular V5
=============================
Standard protocols for all ISN components to ensure strict interchangeability.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, Protocol, runtime_checkable

@runtime_checkable
class ISNComponent(Protocol):
    """Base protocol for all ISN lifecycle components."""
    d_model: int
    d_embedding: int

@runtime_checkable
class ScannerProtocol(ISNComponent, Protocol):
    """Tokens -> Impulses."""
    def forward(
        self, 
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

@runtime_checkable
class WorldEngineProtocol(ISNComponent, Protocol):
    """Impulses -> Emissions."""
    def forward(
        self, 
        impulses: torch.Tensor,
        world_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        ...

@runtime_checkable
class EmitterProtocol(ISNComponent, Protocol):
    """Emissions -> Logits."""
    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        ...
