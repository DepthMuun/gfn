import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..interfaces.loss import Loss

class BaseLoss(nn.Module):
    """
    Base implementation for GFN loss functions.
    Conforms to the Loss protocol.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
