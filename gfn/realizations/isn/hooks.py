"""
ISN Hook System — GFN V4
Middleware for GFN telemetry and visualization.
"""

import torch
from typing import Dict, Any, List, Optional

class ISNHook:
    """
    Base class for ISN lifecycle hooks.
    """
    def before_scanner(self, token_ids: torch.Tensor):
        pass
    
    def after_scanner(self, impulses: torch.Tensor):
        pass
    
    def before_world(self, world_input: Dict[str, Any]):
        pass
    
    def after_world(self, world_output: Dict[str, Any]):
        pass
    
    def on_emission(self, step: int, token_id: int, energy: float):
        pass

class HookManager:
    """
    Manages and executes a list of ISNHooks.
    """
    def __init__(self, hooks: Optional[List[ISNHook]] = None):
        self.hooks = hooks or []

    def before_scanner(self, token_ids):
        for h in self.hooks: h.before_scanner(token_ids)
        
    def after_scanner(self, impulses):
        for h in self.hooks: h.after_scanner(impulses)
        
    def before_world(self, world_input):
        for h in self.hooks: h.before_world(world_input)
        
    def after_world(self, world_output):
        for h in self.hooks: h.after_world(world_output)
        
    def on_emission(self, step, token_id, energy):
        for h in self.hooks: h.on_emission(step, token_id, energy)
