import torch
import torch.nn as nn
from .pooling import HamiltonianPooling
from .hierarchical import HierarchicalAggregator
from .momentum import MomentumAggregator
from ....models.hooks import Plugin, HookManager

__all__ = ['HamiltonianPooling', 'HierarchicalAggregator', 'MomentumAggregator', 'PoolingPlugin']

class PoolingPlugin(Plugin):
    """
    Integrates pooling strategies into the model lifecycle.
    Captures trajectory (x, v) and aggregates it at the end of the batch.
    """
    def __init__(self, pooling_module: nn.Module):
        super().__init__()
        self.pooling = pooling_module
        self.trajectory_x = []
        self.trajectory_v = []

    def register_hooks(self, manager: HookManager):
        manager.register("on_batch_start", self.on_batch_start)
        manager.register("on_timestep_end", self.on_timestep_end)
        manager.register("on_batch_end", self.on_batch_end)

    def on_batch_start(self, **kwargs):
        self.trajectory_x = []
        self.trajectory_v = []

    def on_timestep_end(self, x: torch.Tensor, v: torch.Tensor, **kwargs):
        # x, v are [B, H, D]
        self.trajectory_x.append(x)
        self.trajectory_v.append(v)

    def on_batch_end(self, **kwargs):
        if not self.trajectory_x:
            return None
        
        # [B, L, H, D]
        x_seq = torch.stack(self.trajectory_x, dim=1)
        v_seq = torch.stack(self.trajectory_v, dim=1)
        
        # Flatten heads for standard pooling if needed, or handle per head
        B, L, H, D = x_seq.shape
        x_seq = x_seq.view(B, L, H*D)
        v_seq = v_seq.view(B, L, H*D)
        
        return self.pooling(x_seq, v_seq)
