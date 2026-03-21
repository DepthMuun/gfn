"""
Singularity Components — GFN V5
================================
Migrated and modularized from gfn_original/nn/layers/physics/singularities.py
Provides smooth gating/damping logic to prevent numerical explosions
near manifold singularities where Christoffel symbols diverge.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ...constants import SINGULARITY_THRESHOLD, SINGULARITY_GATE_SLOPE, EPS
from ...config.schema import SingularityConfig


class SingularityGate(nn.Module):
    """
    Manages manifold singularities by providing smooth gating/damping logic.
    
    Prevents numerical explosions in regions where Christoffel symbols diverge
    by applying smooth damping based on proximity to singular values.
    Uses a sigmoid-based gate that smoothly transitions to zero as the metric
    component approaches the singularity threshold.
    
    Args:
        threshold: Distance from singularity at which damping begins (default: 0.5)
        slope: Steepness of the sigmoid gate (higher = sharper transition, default: 10.0)
    """
    
    def __init__(
        self, 
        threshold: float = SINGULARITY_THRESHOLD, 
        slope: float = SINGULARITY_GATE_SLOPE
    ):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
        
    @classmethod
    def from_config(cls, config: SingularityConfig) -> 'SingularityGate':
        """Create SingularityGate from SingularityConfig."""
        if not config.enabled:
            # Return a dummy gate that returns ones (no damping)
            return cls(threshold=0.0, slope=float('inf'))
        return cls(
            threshold=config.threshold,
            slope=config.strength * 20.0  # Scale strength to slope
        )
    
    def forward(
        self, 
        x: Optional[torch.Tensor], 
        v: Optional[torch.Tensor], 
        metric_component: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies smooth damping based on proximity to a singular value.
        
        The gate uses a sigmoid function:
            gate = sigmoid(slope * (dist_to_sing - threshold))
        
        When metric_component approaches 0 (singularity), the gate smoothly
        transitions to 0, damping both velocity and force.
        
        Args:
            x: Position tensor (unused, for API compatibility)
            v: Velocity tensor (unused, for API compatibility)
            metric_component: Tensor representing distance to singularity
        Returns:
            Gate values in range (0, 1)
        """
        # Compute distance to singularity (metric_component -> 0)
        dist_to_sing = torch.abs(metric_component)
        
        # Smooth sigmoid gate
        gate = torch.sigmoid(self.slope * (dist_to_sing - self.threshold))
        
        return gate
    
    def damp_velocity(self, v: torch.Tensor, metric_component: torch.Tensor) -> torch.Tensor:
        """
        Apply damping to velocity tensor.
        
        Args:
            v: Velocity tensor to damp
            metric_component: Tensor representing distance to singularity
        Returns:
            Damped velocity tensor
        """
        gate = self.forward(None, None, metric_component)
        return v * gate
    
    def damp_force(self, force: torch.Tensor, metric_component: torch.Tensor) -> torch.Tensor:
        """
        Apply damping to force tensor.
        
        Args:
            force: Force tensor to damp
            metric_component: Tensor representing distance to singularity
        Returns:
            Damped force tensor
        """
        gate = self.forward(None, None, metric_component)
        return force * gate
    
    def extra_repr(self) -> str:
        return f'threshold={self.threshold}, slope={self.slope}'


class SingularityDetector(nn.Module):
    """
    Detects proximity to manifold singularities based on metric tensor properties.
    
    This component analyzes the metric tensor to identify regions where
    numerical computation may become unstable due to extreme curvature or
    metric degeneracy.
    """
    
    def __init__(
        self,
        threshold: float = SINGULARITY_THRESHOLD,
        epsilon: float = EPS
    ):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        
    def forward(self, metric_tensor: torch.Tensor) -> torch.Tensor:
        """
        Detect singularities by analyzing metric tensor.
        
        Args:
            metric_tensor: Metric tensor of shape [..., D, D]
        Returns:
            Binary mask (1.0 = near singularity, 0.0 = safe)
        """
        # Compute determinant as a measure of metric degeneracy
        # For near-singular metrics, determinant approaches 0
        det = torch.linalg.det(metric_tensor)
        
        # Also check for very small eigenvalues
        eigenvalues = torch.linalg.eigvalsh(metric_tensor)
        min_eigenvalue = eigenvalues.min(dim=-1).values
        
        # Combined singularity measure
        singularity_measure = torch.minimum(
            torch.abs(det),
            torch.abs(min_eigenvalue)
        )
        
        # Binary mask: 1 where singularity is detected
        is_singular = (singularity_measure < self.threshold).float()
        
        return is_singular
    
    def get_metric_component(self, metric_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get a scalar measure of metric component for gating.
        
        Args:
            metric_tensor: Metric tensor of shape [..., D, D]
        Returns:
            Scalar measure (larger = further from singularity)
        """
        # Use determinant as the primary measure
        det = torch.abs(torch.linalg.det(metric_tensor))
        
        # Add small epsilon to avoid log(0)
        det = torch.clamp(det, min=self.epsilon)
        
        return det


class SingularityRegistry:
    """
    Registry for singularity handling strategies.
    
    Provides a factory for creating singularity handlers based on
    configuration and geometry type.
    """
    
    _handlers: Dict[str, type] = {
        'gate': SingularityGate,
        'detector': SingularityDetector,
    }
    
    @classmethod
    def create(cls, config: SingularityConfig, strategy: str = 'gate') -> nn.Module:
        """
        Create a singularity handler based on configuration.
        
        Args:
            config: Singularity configuration
            strategy: Handler strategy ('gate' or 'detector')
        Returns:
            Singularity handler module
        """
        if not config.enabled:
            # Return identity module when disabled
            return nn.Identity()
        
        handler_cls = cls._handlers.get(strategy, SingularityGate)
        
        if strategy == 'gate':
            return handler_cls(
                threshold=config.threshold,
                slope=config.strength * 20.0
            )
        elif strategy == 'detector':
            return handler_cls(
                threshold=config.threshold,
                epsilon=config.epsilon
            )
        
        return handler_cls()
    
    @classmethod
    def register(cls, name: str, handler_cls: type) -> None:
        """Register a custom singularity handler."""
        cls._handlers[name] = handler_cls
