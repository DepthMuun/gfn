import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ...models.hooks import Plugin, HookManager

class LensingPlugin(Plugin):
    """
    Plugin for Geodesic Lensing (Iterative Correction).
    Manages shared Jacobi history across layers to maintain steering stability.
    """
    def __init__(self):
        super().__init__()
        # Store jacobi states indexed by the layer instance
        self.jacobi_states: Dict[nn.Module, Optional[torch.Tensor]] = {}

    def register_hooks(self, manager: HookManager):
        manager.register("on_batch_start", self.on_batch_start)
        manager.register("on_layer_start", self.on_layer_start)
        manager.register("on_layer_end", self.on_layer_end)

    def on_batch_start(self, **kwargs):
        """Reset Jacobi history for new batch."""
        self.jacobi_states = {}

    def on_layer_start(self, layer: nn.Module, layer_kwargs: Dict[str, Any], **kwargs):
        """Inject previous Jacobi state into the layer forward pass."""
        if layer in self.jacobi_states:
            layer_kwargs["jacobi_history"] = self.jacobi_states[layer]

    def on_layer_end(self, layer: nn.Module, extra_info: Dict[str, Any], **kwargs):
        """Collect updated Jacobi state from the layer metadata."""
        if "jacobi_history" in extra_info:
            self.jacobi_states[layer] = extra_info["jacobi_history"]
