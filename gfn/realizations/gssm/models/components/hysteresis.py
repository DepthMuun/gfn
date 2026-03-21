import torch
import torch.nn as nn
from typing import Optional, Dict, Any, cast
from ...models.hooks import Plugin, HookManager
from ...physics.components.hysteresis import HysteresisModule

class HysteresisPlugin(Plugin):
    """
    Plugin for Active Inference / Hysteresis.
    Manages the persistent state 'h' across timesteps and injects Ghost Forces.
    """
    def __init__(self, hysteresis_module: HysteresisModule, 
                 dim: int, heads: int, topology_id: int = 0):
        super().__init__()
        self.module = hysteresis_module
        self.dim = dim
        self.heads = heads
        self.topology_id = topology_id
        self.current_h: Optional[torch.Tensor] = None

    def register_hooks(self, manager: HookManager):
        manager.register("on_batch_start", self.on_batch_start)
        manager.register("on_timestep_start", self.on_timestep_start)
        manager.register("on_batch_end", self.on_batch_end)

    def on_batch_start(self, batch_size: int, device: torch.device, **kwargs):
        """Initializes the hysteresis state at the start of a batch."""
        # Multi-head state [B, H, HiddenDim]
        # HysteresisModule manages its own dimensions, we just need to provide the initial zero tensor
        # or let the module handle it if it supported it (it expects current_h in update_state)
        self.current_h = torch.zeros(
            batch_size, 
            self.heads, 
            self.module.hidden_dim, 
            device=device
        )

    def on_timestep_start(self, x: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Updates memory state and returns the Ghost Force to be added to the impulse.
        """
        # 1. Update internal state
        self.current_h = self.module.update_state(
            self.current_h, x, v, self.topology_id
        )
        
        # 2. Compute ghost force from new state
        ghost_force = self.module.get_ghost_force(self.current_h)
        
        # 3. Return ghost force to BaseModel (which will add it to the step force)
        return ghost_force

    def on_batch_end(self, **kwargs):
        """Cleanup state."""
        self.current_h = None
