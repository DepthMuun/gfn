"""
Hysteresis Components — GFN V5
================================
Migrated and modularized from gfn_original/nn/layers/physics/hysteresis.py
Provides stateful memory dynamics with ghost force readout for manifold evolution.
Decouples stateful hysteresis dynamics from core integrator logic.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict

from ...constants import EPS
from ...config.schema import HysteresisConfig


# Default values from original GFN
DEFAULT_HYSTERESIS_DECAY = 0.95
GHOST_FORCE_SCALE = 0.1


class HysteresisModule(nn.Module):
    """
    Encapsulates Hysteresis (Memory) state management and Ghost Force readout.
    
    Provides stateful memory dynamics that persist across timesteps, enabling
    the model to maintain internal state information. The ghost force readout
    provides corrective forces based on the current hysteresis state.
    
    Supports 3D tensors [Batch, Heads, Dim] for multi-head configurations.
    
    Args:
        dim: Feature dimension
        heads: Number of memory heads (default: 1)
        hidden_dim: Hidden dimension for state update (default: same as dim)
    """
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 1, 
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        
        # State update parameters: Input [x_feat, v]
        # Multi-head weights using 3D parameters [Heads, D_out, D_in]
        # For torus: x_feat has 2*dim (sin/cos encoding)
        # For euclidean: x_feat has dim (no encoding)
        self.update_w = nn.Parameter(
            torch.randn(self.heads, self.hidden_dim, self.dim * 3)
        )
        self.update_b = nn.Parameter(
            torch.zeros(self.heads, self.hidden_dim)
        )
        
        # Readout parameters for ghost force
        self.readout_w = nn.Parameter(
            torch.randn(self.heads, self.dim, self.hidden_dim)
        )
        self.readout_b = nn.Parameter(
            torch.zeros(self.heads, self.dim)
        )
        
        # Registered buffers for state persistence
        self.state: Optional[torch.Tensor]
        self.last_v: Optional[torch.Tensor]
        self.last_x: Optional[torch.Tensor]
        self.register_buffer('state', None)
        self.register_buffer('last_v', None)
        self.register_buffer('last_x', None)
        
        # Initialization
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with appropriate scales."""
        nn.init.orthogonal_(self.update_w, gain=0.1)
        nn.init.orthogonal_(self.readout_w, gain=0.01)  # Small initial ghost force
        
    @classmethod
    def from_config(cls, config: HysteresisConfig, dim: int, heads: int = 1) -> 'HysteresisModule':
        """Create HysteresisModule from HysteresisConfig."""
        if not config.enabled:
            # Return a dummy module that returns zero ghost force
            return None
        
        return cls(
            dim=dim,
            heads=heads,
            hidden_dim=dim  # Use same dimension for hidden state
        )
    
    def _extract_features(self, x: torch.Tensor, topo_id: int) -> torch.Tensor:
        """
        Extract geometric features for memory input.
        
        Args:
            x: Position tensor [..., H, D]
            topo_id: Topology identifier (1 = torus, 0 = euclidean)
        Returns:
            Feature tensor [..., H, 2*D] for torus, [..., H, D] for euclidean
        """
        if topo_id == 1:  # Torus
            # Sinusoidal encoding for toroidal coordinates
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            # Euclidean: pad with zeros to maintain input dimension
            return torch.cat([x, torch.zeros_like(x)], dim=-1)
    
    def update_state(
        self, 
        h_state: Optional[torch.Tensor], 
        x: torch.Tensor, 
        v: torch.Tensor, 
        topo_id: int,
        decay: float = DEFAULT_HYSTERESIS_DECAY
    ) -> torch.Tensor:
        """
        Update the internal hysteresis state.
        
        State update equation:
            h_t = h_{t-1} * decay + tanh(W * [feat(x), v] + b)
        
        Args:
            h_state: Previous hysteresis state [..., H, hidden_dim]
            x: Current position [..., H, D]
            v: Current velocity [..., H, D]
            topo_id: Topology identifier
            decay: Decay factor for previous state (default: 0.95)
        Returns:
            Updated hysteresis state
        """
        x_feat = self._extract_features(x, topo_id)
        
        # Concatenate features with velocity
        h_in = torch.cat([x_feat, v], dim=-1)  # [..., H, D_in]
        
        # Vectorized multi-head matmul
        # h_in: [..., H, D_in], update_w: [H, D_out, D_in]
        # Output: [..., H, D_out]
        update = torch.tanh(
            torch.matmul(h_in.unsqueeze(-2), self.update_w.transpose(-1, -2)).squeeze(-2) + self.update_b
        )
        
        if h_state is None:
            # Initialize state with first update
            return update
            
        # Apply decay to previous state and add new update
        return h_state * decay + update
    
    def get_ghost_force(self, h_state: torch.Tensor) -> torch.Tensor:
        """
        Compute the corrective 'Ghost Force' from the current hysteresis state.
        
        Args:
            h_state: Current hysteresis state [..., H, hidden_dim]
        Returns:
            Ghost force tensor [..., H, D]
        """
        if h_state is None:
            # Return zero force for uninitialized state
            return torch.zeros(
                1, self.heads, self.dim, 
                device=self.state.device if self.state is not None else 'cpu'
            )
        
        # Readout: [..., H, hidden_dim] @ [H, hidden_dim, D] -> [..., H, D]
        force = torch.matmul(
            h_state.unsqueeze(-2), 
            self.readout_w.transpose(-1, -2)
        ).squeeze(-2) + self.readout_b
        
        # Scale ghost force
        return force * GHOST_FORCE_SCALE
    
    def forward(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        topo_id: int = 0
    ) -> torch.Tensor:
        """
        Forward pass updating state and returning ghost force.
        
        Args:
            x: Position tensor [B, H, D] or [B, D]
            v: Velocity tensor [B, H, D] or [B, D]
            topo_id: Topology identifier (default: 0 = euclidean)
        Returns:
            Ghost force tensor
        """
        batch = x.shape[0]
        
        # Handle 2D input (legacy support)
        if x.dim() == 2:
            x_in = x.unsqueeze(1).repeat(1, self.heads, 1) if self.heads > 1 else x.unsqueeze(1)
            v_in = v.unsqueeze(1).repeat(1, self.heads, 1) if self.heads > 1 else v.unsqueeze(1)
        else:
            x_in, v_in = x, v
            
        # Update hysteresis state
        self.state = self.update_state(self.state, x_in, v_in, topo_id)
        self.last_v = v_in
        self.last_x = x_in
        
        # Compute ghost force
        force = self.get_ghost_force(self.state)
        
        # Return appropriate shape
        if x.dim() == 2:
            return force.squeeze(1) if self.heads == 1 else force.mean(dim=1)
        return force
    
    def reset(self):
        """Reset internal hysteresis state."""
        self.state = None
        self.last_v = None
        self.last_x = None
        
    def extra_repr(self) -> str:
        return f'dim={self.dim}, heads={self.heads}, hidden_dim={self.hidden_dim}'


class HysteresisState:
    """
    Lightweight wrapper for hysteresis state management.
    
    Provides a cleaner interface for managing hysteresis state
    without the full nn.Module overhead. Useful for scenarios
    where state needs to be managed externally.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 1,
        device: torch.device = torch.device('cpu')
    ):
        self.dim = dim
        self.heads = heads
        self.device = device
        self.state: Optional[torch.Tensor] = None
        self.last_x: Optional[torch.Tensor] = None
        self.last_v: Optional[torch.Tensor] = None
        
    def reset(self):
        """Reset state to None."""
        self.state = None
        self.last_x = None
        self.last_v = None
        
    def to(self, device: torch.device) -> 'HysteresisState':
        """Move state to device."""
        if self.state is not None:
            self.state = self.state.to(device)
        if self.last_x is not None:
            self.last_x = self.last_x.to(device)
        if self.last_v is not None:
            self.last_v = self.last_v.to(device)
        self.device = device
        return self


class HysteresisRegistry:
    """
    Registry for hysteresis handling strategies.
    
    Provides a factory for creating hysteresis modules based on
    configuration and dimensionality.
    """
    
    _handlers: Dict[str, type] = {
        'module': HysteresisModule,
        'state': HysteresisState,
    }
    
    @classmethod
    def create(
        cls, 
        config: HysteresisConfig, 
        dim: int, 
        heads: int = 1,
        strategy: str = 'module'
    ) -> Optional[nn.Module]:
        """
        Create a hysteresis handler based on configuration.
        
        Args:
            config: Hysteresis configuration
            dim: Feature dimension
            heads: Number of heads
            strategy: Handler strategy ('module' or 'state')
        Returns:
            Hysteresis handler module or None if disabled
        """
        if not config.enabled:
            return None
        
        handler_cls = cls._handlers.get(strategy, HysteresisModule)
        
        if strategy == 'module':
            return handler_cls(dim=dim, heads=heads)
        elif strategy == 'state':
            return handler_cls(dim=dim, heads=heads)
        
        return None
    
    @classmethod
    def register(cls, name: str, handler_cls: type) -> None:
        """Register a custom hysteresis handler."""
        cls._handlers[name] = handler_cls
