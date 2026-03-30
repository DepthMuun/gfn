"""
gfn/realizations/gssm/data/factory.py
DataComponentFactory - Factory for creating data components.

Similar to ModelFactory, creates data components from config.
"""
import torch
from typing import Any, Optional, Callable, Dict

from .replay import TrajectoryReplayBuffer
from .transforms import shift_targets, add_bos_token, pad_sequences, create_attention_mask


class DataComponentFactory:
    """
    Factory for creating data components (replay buffer, transforms).
    
    Usage:
        buffer = DataComponentFactory.create_replay_buffer(config)
        transform = DataComponentFactory.create_transform('shift_targets')
    """
    
    # Registry of transform functions
    _TRANSFORMS: Dict[str, Callable] = {
        'shift_targets': shift_targets,
        'add_bos': add_bos_token,
        'pad_sequences': pad_sequences,
        'attention_mask': create_attention_mask,
    }
    
    @staticmethod
    def create_replay_buffer(
        config: Any,
        capacity: Optional[int] = None,
        dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[TrajectoryReplayBuffer]:
        """
        Create TrajectoryReplayBuffer from config.
        
        Args:
            config: ManifoldConfig or dataclass with replay_buffer settings
            capacity, dim, device, dtype: Optional overrides
            
        Returns:
            TrajectoryReplayBuffer if enabled in config, else None
        """
        # Check if replay buffer is enabled
        replay_cfg = getattr(config, 'replay_buffer', None)
        if replay_cfg is None:
            return None
        
        enabled = getattr(replay_cfg, 'enabled', False)
        if not enabled:
            return None
        
        # Get parameters from config or use overrides
        cap = capacity or getattr(replay_cfg, 'capacity', 10000)
        d = dim or getattr(config, 'dim', 64)
        dev = device or getattr(config, 'device', torch.device('cpu'))
        dt = dtype or getattr(config, 'dtype', torch.float32)
        
        return TrajectoryReplayBuffer(
            capacity=cap,
            dim=d,
            device=dev,
            dtype=dt,
        )
    
    @staticmethod
    def create_transform(name: str) -> Callable:
        """
        Get a transform function by name.
        
        Args:
            name: Transform name ('shift_targets', 'add_bos', 'pad_sequences', 'attention_mask')
            
        Returns:
            Transform function
            
        Raises:
            KeyError: If transform name not found
        """
        if name not in DataComponentFactory._TRANSFORMS:
            available = list(DataComponentFactory._TRANSFORMS.keys())
            raise KeyError(f"Transform '{name}' not found. Available: {available}")
        
        return DataComponentFactory._TRANSFORMS[name]
    
    @staticmethod
    def list_transforms() -> list[str]:
        """List available transform names."""
        return list(DataComponentFactory._TRANSFORMS.keys())
    
    @staticmethod
    def register_transform(name: str, func: Callable) -> None:
        """
        Register a new transform function.
        
        Args:
            name: Transform identifier
            func: Transform function
        """
        if name in DataComponentFactory._TRANSFORMS:
            raise ValueError(f"Transform '{name}' already registered")
        DataComponentFactory._TRANSFORMS[name] = func


# Convenience function matching the API pattern
def create_data_components(config: Any) -> Dict[str, Any]:
    """
    Create all data components from config.
    
    Returns:
        Dict with 'replay_buffer' and 'transforms' keys
    """
    components = {
        'replay_buffer': DataComponentFactory.create_replay_buffer(config),
        'transforms': {},
    }
    
    # Load transforms from config if specified
    transform_list = getattr(config, 'data_transforms', [])
    for transform_name in transform_list:
        try:
            components['transforms'][transform_name] = DataComponentFactory.create_transform(transform_name)
        except KeyError:
            # Log warning but don't fail
            import logging
            logging.warning(f"Transform '{transform_name}' not found, skipping")
    
    return components
