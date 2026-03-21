from typing import Dict, Any, Optional, List
from .base import BaseLoss
from ..registry import LOSS_REGISTRY

# Import to register
from . import generative
from . import physics
from . import toroidal

class LossFactory:
    """Factory for creating GFN loss functions."""
    
    @staticmethod
    def create(config: Any, **kwargs) -> BaseLoss:
        if isinstance(config, str):
            config = {'type': config}
        
        # Merge kwargs into config
        config.update(kwargs)
        
        loss_type = config.get('type', 'generative')
            
        try:
            loss_cls = LOSS_REGISTRY.get(loss_type)
            return loss_cls(config)
        except KeyError:
            print(f"Warning: Loss '{loss_type}' not found, falling back to generative.")
            from .generative import ManifoldGenerativeLoss
            return ManifoldGenerativeLoss(config)
            
    @staticmethod
    def create_multitask(configs: List[Dict[str, Any]]) -> List[BaseLoss]:
        return [LossFactory.create(cfg) for cfg in configs]
