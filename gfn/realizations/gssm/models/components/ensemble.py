import torch
from typing import Tuple, Dict, Any, Optional
from ...models.hooks import Plugin, HookManager

class EnsemblePlugin(Plugin):
    """
    Plugin for Multi-Trajectory Geodesic Flow (MTGF).
    Handles force broadcasting for ensemble mode.
    """
    def __init__(self, n_trajectories: int = 1):
        super().__init__()
        self.n_trajectories = n_trajectories

    def register_hooks(self, manager: HookManager):
        manager.register("on_resolve_forces", self.on_resolve_forces)

    def on_resolve_forces(self, all_forces: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Broadcasts forces and masks to the ensemble dimension: [B, L, D] -> [B, L, H, D]."""
        if self.n_trajectories > 1:
            # We assume the model expects [B, L, H, HD] 
            # all_forces is typically [B, L, D] (where D is total dim)
            # or [B, L, HD] if already projected. 
            # In V5, force passed to layers should be per-head if heads > 1.
            all_forces = all_forces.unsqueeze(2).expand(-1, -1, self.n_trajectories, -1)
            mask = mask.unsqueeze(2).expand(-1, -1, self.n_trajectories, -1)
        return all_forces, mask
