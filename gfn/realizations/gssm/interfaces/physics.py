from typing import Protocol, runtime_checkable, Dict, Any, Optional
from torch import Tensor

@runtime_checkable
class PhysicsEngine(Protocol):
    """Protocol for the core physics engine."""
    
    def compute_acceleration(self, x: Tensor, v: Tensor, force: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """Computes total acceleration given state and external force."""
        ...
        
    def validate_state(self, x: Tensor, v: Tensor) -> None:
        """Validates that state tensors are within physical boundaries."""
        ...
