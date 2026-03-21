from typing import Protocol, runtime_checkable, Any, Dict
from torch import Tensor

@runtime_checkable
class Integrator(Protocol):
    """Protocol for numerical integrators."""
    
    def step(self, x: Tensor, v: Tensor, forces: Tensor, dt: float, **kwargs) -> Dict[str, Tensor]:
        """
        Performs a single integration step.
        Returns a dictionary containing the new state (x, v) and optional diagnostic info.
        """
        ...
