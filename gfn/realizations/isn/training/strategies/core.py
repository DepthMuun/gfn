"""
Core training strategies including full BPTT, truncated BPTT, and STE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from ...registry import strategies

class BaseStrategy:
    """Base logic for strategies."""
    requires_chunking: bool = False
    chunk_size: int = 0
    
    def prepare_model(self, model: nn.Module) -> None:
        pass
    
    def post_backward_hook(self, model: nn.Module) -> None:
        pass

@strategies.register("full")
@strategies.register("full_bptt")
class FullBPTT(BaseStrategy):
    """O(L) memory standard BPTT."""
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {'loss': loss}

@strategies.register("tbptt")
@strategies.register("truncated")
class TruncatedBPTT(BaseStrategy):
    """O(k2) memory truncated BPTT."""
    def __init__(self, k2: int = 64, **kwargs):
        self.requires_chunking = True
        self.chunk_size = k2
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {'loss': loss}
    
    def post_backward_hook(self, model: nn.Module) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

@strategies.register("ste")
@strategies.register("straight_through")
class StraightThroughEstimator(BaseStrategy):
    """O(1) memory gradient approximation (STE)."""
    def __init__(self, gradient_scale: float = 1.0, **kwargs):
        self.gradient_scale = gradient_scale

    def prepare_model(self, model: nn.Module) -> None:
        # We assume the world has a 'use_ste' flag
        for module in model.modules():
            if hasattr(module, 'use_ste'):
                module.use_ste = True

    class _STEFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, impulse, flow_gate_fn):
            new_state = torch.tanh(flow_gate_fn(state + impulse))
            return new_state
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output, None

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {'loss': loss * self.gradient_scale}
