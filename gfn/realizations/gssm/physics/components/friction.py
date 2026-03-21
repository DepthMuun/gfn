"""
Friction Components — GFN V5
=============================
Modular friction/damping components for manifold physics.
Provides both static and learned friction gates.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ...constants import DEFAULT_FRICTION, GATE_BIAS_CLOSED
from ...config.schema import StabilityConfig


class FrictionGate(nn.Module):
    """
    Configurable friction/damping gate for manifold dynamics.
    
    Provides multiple friction modes:
    - 'static': Fixed friction coefficient
    - 'mlp': Learned friction based on position/velocity features
    
    Args:
        dim: Feature dimension
        gate_input_dim: Input dimension for MLP gate (default: dim * 2 for sin/cos encoding)
        mode: Friction mode ('static' or 'mlp')
        num_heads: Number of heads for multi-head configuration
    """
    
    def __init__(
        self, 
        dim: int, 
        gate_input_dim: Optional[int] = None,
        mode: str = 'static', 
        num_heads: int = 1
    ):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.dim = dim
        
        if gate_input_dim is None:
            gate_input_dim = dim * 2  # Default: sin/cos encoding
            
        if mode == 'mlp':
            if num_heads > 1:
                # Multi-head learned friction
                self.gate_w = nn.Parameter(torch.zeros(num_heads, gate_input_dim, dim))
                self.gate_b = nn.Parameter(torch.zeros(num_heads, dim))
            else:
                # Single-head MLP
                self.gate_fc = nn.Linear(gate_input_dim, dim)
                nn.init.zeros_(self.gate_fc.weight)
                nn.init.zeros_(self.gate_fc.bias)
                
        # Static friction parameter
        self.static_friction = nn.Parameter(torch.full((1,), DEFAULT_FRICTION))
        
    @classmethod
    def from_config(cls, config: StabilityConfig, dim: int, num_heads: int = 1) -> 'FrictionGate':
        """Create FrictionGate from StabilityConfig."""
        mode = config.friction_mode if hasattr(config, 'friction_mode') else 'static'
        return cls(dim=dim, mode=mode, num_heads=num_heads)
        
    def forward(
        self, 
        x_in: torch.Tensor, 
        force: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute friction coefficient.
        
        Args:
            x_in: Input features for MLP gate (position encoding)
            force: Optional external force for additional context
        Returns:
            Friction coefficient tensor
        """
        if self.mode == 'mlp':
            if self.num_heads > 1:
                # Multi-head: [..., H, gate_input_dim] @ [H, gate_input_dim, D] -> [..., H, D]
                return torch.sigmoid(
                    torch.matmul(x_in.unsqueeze(-2), self.gate_w).squeeze(-2) + self.gate_b
                )
            return torch.sigmoid(self.gate_fc(x_in))
            
        # Static mode: return constant friction
        return torch.abs(self.static_friction)
    
    def extra_repr(self) -> str:
        return f'mode={self.mode}, num_heads={self.num_heads}, dim={self.dim}'


class AdaptiveFriction(nn.Module):
    """
    Adaptive friction that adjusts based on velocity magnitude.
    
    Provides velocity-dependent friction for more stable dynamics,
    especially useful for preventing velocity explosions.
    """
    
    def __init__(
        self,
        base_friction: float = DEFAULT_FRICTION,
        velocity_scale: float = 0.1,
        clamp_velocity: float = 10.0
    ):
        super().__init__()
        self.base_friction = base_friction
        self.velocity_scale = velocity_scale
        self.clamp_velocity = clamp_velocity
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive friction coefficient.
        
        Args:
            v: Velocity tensor
        Returns:
            Adaptive friction coefficient
        """
        # Compute velocity magnitude
        v_mag = torch.norm(v, dim=-1, keepdim=True)
        v_mag = torch.clamp(v_mag, max=self.clamp_velocity)
        
        # Adaptive: friction increases with velocity
        friction = self.base_friction * (1.0 + self.velocity_scale * v_mag)
        
        return friction
    
    def extra_repr(self) -> str:
        return f'base_friction={self.base_friction}, velocity_scale={self.velocity_scale}'


class FrictionRegistry:
    """
    Registry for friction handling strategies.
    
    Provides a factory for creating friction modules based on
    configuration.
    """
    
    _handlers: Dict[str, type] = {
        'static': FrictionGate,
        'mlp': FrictionGate,
        'adaptive': AdaptiveFriction,
    }
    
    @classmethod
    def create(
        cls, 
        config: StabilityConfig, 
        dim: int, 
        num_heads: int = 1,
        mode: Optional[str] = None
    ) -> nn.Module:
        """
        Create a friction handler based on configuration.
        
        Args:
            config: Stability configuration
            dim: Feature dimension
            num_heads: Number of heads
            mode: Override friction mode
        Returns:
            Friction handler module
        """
        friction_mode = mode or getattr(config, 'friction_mode', 'static')
        
        if friction_mode == 'adaptive':
            return AdaptiveFriction(
                base_friction=config.friction,
                velocity_scale=config.velocity_friction_scale
            )
            
        handler_cls = cls._handlers.get(friction_mode, FrictionGate)
        return handler_cls(
            dim=dim, 
            mode=friction_mode, 
            num_heads=num_heads
        )
    
    @classmethod
    def register(cls, name: str, handler_cls: type) -> None:
        """Register a custom friction handler."""
        cls._handlers[name] = handler_cls
