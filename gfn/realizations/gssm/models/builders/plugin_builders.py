"""
gfn/realizations/gssm/models/builders/plugin_builders.py
Builders for optional model plugins (pooling, checkpointing, adjoint).
"""
import torch.nn as nn
from typing import Any, Optional

from . import ComponentBuilder, MODEL_BUILDER_REGISTRY
from ..components.pooling import HamiltonianPooling, HierarchicalAggregator, MomentumAggregator, PoolingPlugin


class PoolingBuilder(ComponentBuilder):
    """Builder for optional pooling plugin."""
    
    def __init__(self, config: Any, topology: str):
        super().__init__(config)
        self.topology = topology
    
    def build(self) -> Optional[nn.Module]:
        """Build pooling plugin if enabled, otherwise return None."""
        pooling_type = getattr(self.config, 'pooling_type', None)
        
        if not pooling_type:
            return None
        
        if pooling_type == 'hamiltonian':
            pool_mod = HamiltonianPooling(self.config.dim, topology_type=self.topology)
        elif pooling_type == 'hierarchical':
            pool_mod = HierarchicalAggregator(self.config.dim, topology_type=self.topology)
        elif pooling_type == 'momentum':
            pool_mod = MomentumAggregator(self.config.dim, topology_type=self.topology)
        else:
            return None
        
        return PoolingPlugin(pool_mod)


class CheckpointingBuilder(ComponentBuilder):
    """Builder for optional checkpointing plugin."""
    
    def build(self) -> Optional[nn.Module]:
        """Build checkpointing plugin if enabled, otherwise return None."""
        ckpt_cfg = self.config.physics.checkpointing
        
        if not ckpt_cfg.get('enabled', False):
            return None
        
        from ..components.checkpointing import CheckpointingPlugin
        
        return CheckpointingPlugin(
            chunk_size=ckpt_cfg.get('chunk_size', 32)
        )


class AdjointBuilder(ComponentBuilder):
    """Builder for optional adjoint plugin."""
    
    def build(self) -> Optional[nn.Module]:
        """Build adjoint plugin if enabled, otherwise return None."""
        if not getattr(self.config, 'adjoint_enabled', False):
            return None
        
        from ..components.adjoint import AdjointPlugin
        
        return AdjointPlugin(self.config)


# Register the builders
MODEL_BUILDER_REGISTRY.register('pooling', PoolingBuilder)
MODEL_BUILDER_REGISTRY.register('checkpointing', CheckpointingBuilder)
MODEL_BUILDER_REGISTRY.register('adjoint', AdjointBuilder)
