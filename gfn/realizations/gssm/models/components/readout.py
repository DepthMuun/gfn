import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from ...interfaces.model import GFNModel
from ...constants import TOPOLOGY_TORUS
from ...models.hooks import Plugin, HookManager

class BaseReadout(nn.Module):
    """Abstract base for all readout strategies."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class IdentityReadout(BaseReadout):
    """Holographic Readout: Returns the manifold state directly.
    WARNING: This expects a latent loss (like MSE or Toroidal). 
    Using NLL with this will crash if ManifoldDim != VocabSize.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] or [B, H, HD]
        # Always return [B, D]
        if x.dim() == 3:
            return x.flatten(1)
        return x

class CategoricalReadout(BaseReadout):
    """Standard Readout for classification tasks (Maps to Vocab)."""
    def __init__(self, dim: int, vocab_size: int, topology_type: str = 'euclidean', gain: float = 2.0):
        super().__init__()
        self.topology_type = topology_type
        # For toroidal, we project to [sin(x), cos(x)]
        in_dim = dim * 2 if topology_type == TOPOLOGY_TORUS else dim
        self.linear = nn.Linear(in_dim, vocab_size)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] or [B, H, HD]
        if x.dim() == 3:
            x = x.flatten(1)
            
        # Robust comparison using constants to avoid hardcoding
        is_torus = str(self.topology_type).lower().strip() == TOPOLOGY_TORUS
        
        if is_torus:
            x_feat = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            x_feat = x
            
        return self.linear(x_feat)

class ImplicitReadout(BaseReadout):
    """
    Implicit Readout: Maps latent state to a target coordinate space via MLP.
    Useful for regression or high-dimensional latent alignment.
    """
    def __init__(self, dim: int, out_dim: int, hidden_dim: int = 128, 
                 topology_type: str = 'euclidean'):
        super().__init__()
        self.topology_type = topology_type
        in_dim = dim * 2 if topology_type == TOPOLOGY_TORUS else dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(1)
        is_torus = str(self.topology_type).lower().strip() == TOPOLOGY_TORUS
        if is_torus:
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.net(x)

class ReadoutPlugin(Plugin):
    """Wraps a Readout module and hooks it into the model lifecycle."""
    def __init__(self, readout_module: BaseReadout):
        super().__init__()
        self.readout = readout_module

    def register_hooks(self, manager: HookManager):
        manager.register("on_timestep_end", self.on_timestep_end)

    def on_timestep_end(self, x: torch.Tensor, **kwargs):
        """Processes the state x through the readout and returns logits."""
        return self.readout(x)
