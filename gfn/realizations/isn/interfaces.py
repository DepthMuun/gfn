"""
ISN Interfaces — GFN V4
Base abstract classes for component interchangeability.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

class BaseScanner(nn.Module, ABC):
    """
    Interface for the input processor (Encoder).
    Tokens -> Impulses (Kinetic Geometric Entries).
    """
    @abstractmethod
    def forward(
        self, 
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            token_ids: (batch, seq_len)
            state: (batch, d_model) or similar
        Returns:
            impulses: (batch, seq_len, d_model)
            final_state: (batch, d_model)
        """
        pass

class BaseWorldEngine(nn.Module, ABC):
    """
    Interface for the topological simulator.
    Impulses -> Raw Emissions & World State.
    """
    @abstractmethod
    def forward(
        self, 
        impulses: torch.Tensor,
        noise_std: float = 0.0,
        max_burst: int = 5,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            impulses: (batch, seq_len, d_model)
        Returns:
            Dict with mandatory keys: 'emitted_embeddings' (batch, seq_len_out, d_embedding).
        """
        pass

class BaseEmitter(nn.Module, ABC):
    """
    Interface for materialization/collapsing logic.
    Raw World Data -> Final Predictions/Physical Outcomes.
    """
    @abstractmethod
    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emitted_embeddings: (batch, seq_len_out, d_embedding)
        Returns:
            logits: (batch, seq_len_out, vocab_size)
        """
        pass
