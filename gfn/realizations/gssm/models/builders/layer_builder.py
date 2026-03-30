"""
gfn/realizations/gssm/models/builders/layer_builder.py
Builder for ManifoldLayer components.
"""
import torch.nn as nn
from typing import Any, Tuple

from . import ComponentBuilder, MODEL_BUILDER_REGISTRY
from ..manifold_layer import ManifoldLayer
from ..components.mixer import FlowMixer, GeodesicAttentionMixer
from ...geometry.factory import GeometryFactory
from ...physics.engine import ManifoldPhysicsEngine
from ...physics.integrators.factory import IntegratorFactory


class LayerBuilder(ComponentBuilder):
    """
    Builder for the stack of ManifoldLayer components.
    
    Creates all layers with their individual geometries, physics engines,
    integrators, and mixers.
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        # Pre-compute dimensions needed for all layers
        self._compute_dimensions()
    
    def _compute_dimensions(self) -> None:
        """Compute head_dim and other dimensions from config."""
        config = self.config
        
        topology_cfg = config.physics.topology
        geometry_scope = getattr(topology_cfg, 'geometry_scope', 'local')
        
        if geometry_scope == 'global':
            # GDG Mode: Each head has the full dim D
            self.head_dim = config.dim
        else:
            # Local Mode: Heads partition the dim D
            self.head_dim = config.dim // config.heads
        
        self.dim_total = config.heads * self.head_dim
        self.topology = topology_cfg.type
        self.dynamics_type = config.dynamics_type
        self.mixer_type = getattr(config, 'mixer_type', 'low_rank')
    
    def build(self) -> nn.ModuleList:
        """Build and return the list of ManifoldLayer components."""
        layers = nn.ModuleList()
        
        for layer_idx in range(self.config.depth):
            layer = self._build_single_layer(layer_idx)
            layers.append(layer)
        
        return layers
    
    def _build_single_layer(self, layer_idx: int) -> ManifoldLayer:
        """Build a single ManifoldLayer."""
        config = self.config
        
        # Create geometry, physics engine, and integrator for this layer
        geometry = GeometryFactory.create_with_dim(
            self.head_dim, config.rank, config.heads, config.physics
        )
        physics_engine = ManifoldPhysicsEngine(
            geometry, config.physics, dim=self.head_dim, heads=config.heads
        )
        integrator = IntegratorFactory.create(physics_engine, config.physics)
        
        # Create mixer for this layer
        mixer = self._build_mixer()
        
        # Create the layer
        layer = ManifoldLayer(
            integrator=integrator,
            mixer=mixer,
            config=config.physics,
            heads=config.heads,
            head_dim=self.head_dim,
            dynamics_type=self.dynamics_type,
            layer_idx=layer_idx,
            total_depth=config.depth,
        )
        
        return layer
    
    def _build_mixer(self) -> nn.Module:
        """Build the mixer component based on mixer_type."""
        if self.mixer_type == 'attention':
            return GeodesicAttentionMixer(
                self.dim_total, self.config.heads, topology=self.topology
            )
        else:
            return FlowMixer(
                dim=self.dim_total,
                rank=self.config.rank,
                heads=self.config.heads,
                topology=self.topology,
                mode=self.mixer_type,
                use_norm=self.config.physics.stability.enable_trace_normalization,
            )
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Return (head_dim, dim_total) for use by other builders."""
        return self.head_dim, self.dim_total


# Register the builder
MODEL_BUILDER_REGISTRY.register('layers', LayerBuilder)
