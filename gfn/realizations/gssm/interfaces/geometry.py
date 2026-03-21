from typing import Protocol, runtime_checkable, Optional, Tuple, Union
import torch
from torch import Tensor

@runtime_checkable
class Geometry(Protocol):
    """Protocol for Riemannian geometries."""
    
    def metric_tensor(self, x: Tensor) -> Tensor:
        """Computes the metric tensor g_ij at position x."""
        ...
        
    def christoffel_symbols(self, x: Tensor) -> Tensor:
        """Computes the Christoffel symbols Gamma^k_ij at position x."""
        ...
        
    def forward(self, x: Tensor, v: Optional[Tensor] = None, force: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Computes the acceleration (geodesic + external forces).
        May return a single tensor or a tuple (acceleration, friction).
        """
        ...
        
    def project(self, x: Tensor) -> Tensor:
        """Projects a point back to the manifold manifold."""
        ...

    def dist(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Computes the geodesic distance between two points."""
        ...
