"""
Proxy-based training strategy that bypasses recurrent backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from ...registry import strategies

class DirectProjectionProxy(nn.Module):
    """O(1) Proxy Network."""
    def __init__(self, d_model: int, d_embedding: int, n_layers: int = 2):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.aggregator = nn.Sequential(
            nn.Linear(d_model, d_embedding * 2),
            nn.Tanh(),
            nn.Linear(d_embedding * 2, d_embedding)
        )
    def forward(self, impulses: torch.Tensor) -> torch.Tensor:
        # O(1) aggregation
        aggregated = self.input_norm(impulses.sum(dim=1))
        return self.aggregator(aggregated)

@strategies.register("proxy")
@strategies.register("direct_projection")
class ProxyStrategy:
    """Strategy that uses a proxy network to bypass recurrence during backprop."""
    requires_chunking = False
    
    def __init__(self, n_layers: int = 2, **kwargs):
        self.n_layers = n_layers
        self.proxy: Optional[DirectProjectionProxy] = None

    def prepare_model(self, model: nn.Module) -> None:
        # Initialize proxy based on model dimensions
        self.proxy = DirectProjectionProxy(
            d_model=model.d_model, 
            d_embedding=model.d_embedding,
            n_layers=self.n_layers
        ).to(next(model.parameters()).device)
        
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Standard CrossEntropy on the logits emitted by the model
        # Note: In proxy mode, the logits often come from proxy(impulses) -> emitter
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {'loss': loss}

    def post_backward_hook(self, model: nn.Module) -> None:
        pass
