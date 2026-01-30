"""
Manifold State Management
=========================

State container and utilities for Manifold model.
"""

import torch
from typing import Tuple, Optional
from ..exceptions import InvalidStateError


class ManifoldState:
    """State container for Manifold model (x, v) tuples.
    
    This class provides a clean interface for managing position (x) and
    velocity (v) tensors, with validation and utility methods.
    
    Attributes:
        x: Position tensor [batch, dim]
        v: Velocity tensor [batch, dim]
    """
    
    def __init__(self, x: torch.Tensor, v: torch.Tensor):
        """Initialize state with position and velocity tensors.
        
        Args:
            x: Position tensor [batch, dim]
            v: Velocity tensor [batch, dim]
        """
        self.x = x
        self.v = v
    
    @classmethod
    def from_parameters(cls, x0: torch.nn.Parameter, v0: torch.nn.Parameter, 
                       batch_size: int) -> 'ManifoldState':
        """Create state from model parameters.
        
        Args:
            x0: Initial position parameter [1, dim]
            v0: Initial velocity parameter [1, dim]
            batch_size: Batch size to expand to
            
        Returns:
            ManifoldState instance
        """
        x = x0.expand(batch_size, -1)
        v = v0.expand(batch_size, -1)
        return cls(x, v)
    
    @classmethod
    def from_tuple(cls, state_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]],
                   x0: torch.nn.Parameter, v0: torch.nn.Parameter,
                   batch_size: int) -> 'ManifoldState':
        """Create state from tuple or initialize from parameters.
        
        Args:
            state_tuple: Optional (x, v) tuple or None
            x0: Initial position parameter for initialization
            v0: Initial velocity parameter for initialization
            batch_size: Batch size
            
        Returns:
            ManifoldState instance
            
        Raises:
            InvalidStateError: If state_tuple is invalid
        """
        if state_tuple is None:
            return cls.from_parameters(x0, v0, batch_size)
        
        # Validate tuple
        if not isinstance(state_tuple, tuple) or len(state_tuple) != 2:
            raise InvalidStateError(
                f"state must be tuple of (x, v), got {type(state_tuple)}"
            )
        
        x, v = state_tuple
        return cls(x, v)
    
    def validate(self, expected_batch: int, expected_dim: int):
        """Validate state tensor shapes.
        
        Args:
            expected_batch: Expected batch size
            expected_dim: Expected dimension
            
        Raises:
            InvalidStateError: If shapes don't match
        """
        if self.x.shape != (expected_batch, expected_dim):
            raise InvalidStateError(
                f"Invalid x shape: expected ({expected_batch}, {expected_dim}), "
                f"got {self.x.shape}"
            )
        if self.v.shape != (expected_batch, expected_dim):
            raise InvalidStateError(
                f"Invalid v shape: expected ({expected_batch}, {expected_dim}), "
                f"got {self.v.shape}"
            )
    
    def to_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tuple for backward compatibility.
        
        Returns:
            Tuple of (x, v)
        """
        return (self.x, self.v)
    
    def detach(self) -> 'ManifoldState':
        """Detach tensors from computation graph.
        
        Returns:
            New ManifoldState with detached tensors
        """
        return ManifoldState(self.x.detach(), self.v.detach())
    
    def clone(self) -> 'ManifoldState':
        """Clone state tensors.
        
        Returns:
            New ManifoldState with cloned tensors
        """
        return ManifoldState(self.x.clone(), self.v.clone())
    
    @property
    def device(self) -> torch.device:
        """Get device of state tensors."""
        return self.x.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of state tensors."""
        return self.x.dtype
    
    def __repr__(self) -> str:
        return f"ManifoldState(x.shape={self.x.shape}, v.shape={self.v.shape})"
