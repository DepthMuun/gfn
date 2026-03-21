from typing import Protocol, runtime_checkable, Dict, Any, Optional, Tuple, Union
from torch import Tensor
import torch.nn as nn

@runtime_checkable
class GFNModel(Protocol):
    """Protocol for the main GFN model architecture."""
    
    def forward(self, input_ids: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Main forward pass evolving states through the manifold."""
        ...
        
    def step(self, x: Tensor, v: Tensor, impulse: Optional[Tensor] = None, dt: float = 0.1) -> Tuple[Tensor, Tensor]:
        """Single step of state evolution."""
        ...
