import torch
import torch.nn as nn
from typing import List, Any, Dict, Callable, Optional

class HookManager:
    """
    Manages lifecycle hooks for MANIFOLD models.
    Allows external components to inject logic into the forward pass.
    """
    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {
            "pre_forward": [],
            "state_init": [],
            "wrap_evolution": [],
            "on_batch_start": [],
            "on_timestep_start": [],
            "on_layer_start": [],
            "on_layer_end": [],
            "on_timestep_end": [],
            "on_batch_end": []
        }

    def register(self, hook_name: str, callback: Callable):
        if hook_name not in self._hooks:
            raise ValueError(f"Unknown hook: {hook_name}")
        self._hooks[hook_name].append(callback)

    def trigger(self, hook_name: str, *args, **kwargs) -> Any:
        """Triggers all registered callbacks for a specific hook."""
        results = []
        for callback in self._hooks.get(hook_name, []):
            res = callback(*args, **kwargs)
            if res is not None:
                results.append(res)
        return results

class Plugin(nn.Module):
    """Base class for all GFN plugins."""
    def register_hooks(self, manager: HookManager):
        raise NotImplementedError("Plugins must implement register_hooks")
