"""
ISN Threshold Emitter Component — GFN V4
Physics-based materialization.
"""

import torch
import torch.nn as nn
from ...interfaces import BaseEmitter

class ThresholdEmitter(BaseEmitter):
    """
    Standard ISN Emitter based on energy thresholds.
    """
    def __init__(self, d_embedding: int, vocab_size: int, threshold_base: float = 0.5):
        super().__init__()
        self.d_embedding = d_embedding
        self.vocab_size = vocab_size
        
        # Physical decision projections
        self.pre_threshold = nn.Linear(d_embedding, 1, bias=True)
        self.emission = nn.Linear(d_embedding, vocab_size, bias=True)
        
        # Learned energy threshold
        self.threshold = nn.Parameter(torch.ones(1) * threshold_base)

    def forward(self, emitted_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits from emitted embeddings.
        """
        # Linear projection to semantic space
        logits = self.emission(emitted_embeddings)
        return logits
