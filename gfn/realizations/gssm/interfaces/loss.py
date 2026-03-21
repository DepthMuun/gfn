from typing import Protocol, runtime_checkable, Dict, Any
from torch import Tensor

@runtime_checkable
class Loss(Protocol):
    """Protocol for GFN loss functions."""
    
    def forward(self, x_pred: Tensor, x_target: Tensor, **kwargs) -> Tensor:
        """Computes the loss scalar."""
        ...
