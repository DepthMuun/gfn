"""
gfn/models/factory.py — GFN V5
ModelFactory: builds complete ManifoldModel from ManifoldConfig.

Configuration support via:
  - ManifoldConfig directly (config=...)
  - Preset + flat overrides: gfn.create(preset_name='stable-torus', dim=64, ...)
  - Preset + physics dict: gfn.create(preset_name='...', physics={'stability': {'base_dt': 0.5}})
  - Pure physics dict without preset: gfn.create(config=ManifoldConfig(physics=dict_to_physics_config({...})))
"""
import torch
import torch.nn as nn
import logging
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from .manifold import ManifoldModel
from .components.embedding import FunctionalEmbedding
from .components.mixer import FlowMixer, GeodesicAttentionMixer
from .components.readout import CategoricalReadout, ReadoutPlugin, IdentityReadout, ImplicitReadout
from .components.pooling import HamiltonianPooling, HierarchicalAggregator, MomentumAggregator, PoolingPlugin
from ..geometry.factory import GeometryFactory
from ..physics.integrators.factory import IntegratorFactory
from ..physics.engine import ManifoldPhysicsEngine
from ..config.schema import ManifoldConfig, PhysicsConfig
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from ..config.loader import dict_to_physics_config, apply_physics_overrides
from ..registry import MODEL_REGISTRY
from ..errors import ConfigurationError
from ..config.normalizer import normalize_config
from .builders import (
    EmbeddingBuilder, LayerBuilder, ReadoutBuilder,
    PoolingBuilder, CheckpointingBuilder, AdjointBuilder, LensingBuilder
)

from ..config.serialization import from_dict


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory to build GFN V5 models.

    IMPORTANT: Geometry operates on per-head tensors [B, H, HD] where HD = dim/heads.
    The factory passes head_dim to GeometryFactory, not the total dim.

    Supported creation flows:
      1. ModelFactory.create(config=ManifoldConfig(...))
      2. ModelFactory.create(vocab_size=100, dim=64, ...)
      3. ModelFactory.from_pretrained('path/to/model')
    """

    @staticmethod
    def _recursive_setattr(obj, attr_path, value):
        attrs = attr_path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)

    @staticmethod
    def _collect_explicit_keys_from_mapping(mapping: Dict[str, Any], prefix: str = "") -> set:
        """
        Flatten nested config dictionaries into explicit dotted paths.

        Example:
            {"physics": {"topology": {"riemannian_type": "low_rank"}}}
        becomes:
            {"physics", "physics.topology", "physics.topology.riemannian_type"}
        """
        keys = set()
        for k, v in mapping.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict):
                keys.update(ModelFactory._collect_explicit_keys_from_mapping(v, prefix=full_key))
        return keys

    @staticmethod
    def create(
        config: Optional[ManifoldConfig] = None,
        preset_name: Optional[str] = None,
        physics: Optional[Union[Dict[str, Any], PhysicsConfig]] = None,
        **kwargs
    ) -> ManifoldModel:
        """
        Builds a ManifoldModel.

        Args:
            config:      Complete ManifoldConfig. If provided, takes priority.
            preset_name: (DEPRECATED) Physics preset name.
            physics:     Nested dict or PhysicsConfig to override physics.
            **kwargs:    Flat overrides of ManifoldConfig/PhysicsConfig.
                         Supports prefixes to reach nested levels:
                         - 'topology_type', 'base_dt', 'friction', 'integrator'
        """
        # ── 0. Resolve base configuration ────────────────────────────────────
        if isinstance(config, str):
            if config.lower() == 'gssm':
                config = None
            else:
                preset_name = config
                config = None
                
        # Keep track of explicitly provided kwargs to avoid heuristic-only sync
        explicit_keys = set(kwargs.keys())
        
        if preset_name is not None:
             logger.warning("preset_name is deprecated and will be ignored. Use direct configuration or physics overrides.")

        if isinstance(config, dict):
            # Handle potential double wrapping ('config' -> 'architecture'/'physics')
            if 'config' in config and isinstance(config['config'], dict):
                # If it's a full checkpoint dict, we might have 'config' as a key
                config = config['config']

            # Handle potential 'architecture' or 'model' wrapper in legacy or external configs
            for wrapper in ['architecture', 'model']:
                if wrapper in config:
                    wrapped = config.pop(wrapper)
                    if isinstance(wrapped, dict):
                        # Merge fields into the top-level dict (priority to wrapped fields)
                        for k, v in wrapped.items():
                            if k not in config:
                                config[k] = v
                                # If we are loading from a dict, these are considered explicit
                                explicit_keys.add(k)
            
            # Also add any other top-level keys as explicit
            explicit_keys.update(config.keys())
            explicit_keys.update(ModelFactory._collect_explicit_keys_from_mapping(config))
            
            config = from_dict(ManifoldConfig, config)

        # Re-sync with extra kwargs (they take priority over dict values if provided twice)
        explicit_keys.update(kwargs.keys())

        if config is None:
            vsize = kwargs.pop('vocab_size', 100)
            # Initialize with professional defaults
            config = ManifoldConfig(vocab_size=vsize)

        # ── 1. Apply Physics Overrides (Dict/Config) ─────────────────────────
        if physics is not None:
            if isinstance(physics, dict):
                explicit_keys.update(ModelFactory._collect_explicit_keys_from_mapping({'physics': physics}))
                apply_physics_overrides(config.physics, physics)
            elif isinstance(physics, PhysicsConfig):
                config.physics = physics
            else:
                raise ConfigurationError(f"physics must be a dict or PhysicsConfig, got {type(physics)}")

        # ── 2. Normalize Configuration (using ConfigNormalizer) ────────────────
        remaining_kwargs, validation_errors = normalize_config(config, kwargs, explicit_keys)
        if validation_errors:
            logger.warning(f"Config validation warnings: {validation_errors}")
        
        # Remaining kwargs were not mapped (possibly unknown arguments)
        if remaining_kwargs:
            logger.debug(f"Unmapped kwargs: {list(remaining_kwargs.keys())}")

        # Preserve which keys were explicitly requested so downstream builders/factories
        # can distinguish user intent from schema defaults.
        config._explicit_keys = explicit_keys
        config.physics._explicit_keys = explicit_keys

        topology_cfg = config.physics.topology
        geometry_scope = getattr(topology_cfg, 'geometry_scope', 'local')
        
        if geometry_scope == 'global':
            # GDG Mode: Each head has the full dim D. Total state is H * D.
            head_dim = config.dim
        else:
            # Local Mode: Heads partition the dim D. Total state is D.
            head_dim = config.dim // config.heads
            
        # ── 4. Build Components using Builders ─────────────────────────────
        
        # Build embedding
        embedding_builder = EmbeddingBuilder(config)
        embedding = embedding_builder.build()
        
        # Build layers (and get dimensions)
        layer_builder = LayerBuilder(config)
        layers = layer_builder.build()
        head_dim, dim_total = layer_builder.get_dimensions()
        
        topology = config.physics.topology.type
        
        # ── 5. Initial state ─────────────────────────────────────────────────
        spread = getattr(config, 'initial_spread', 0.1)
        x0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
        v0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
        
        # ── 6. Model assembly ────────────────────────────────────────────────
        store_full = getattr(config, 'store_full_sequence', True)
        model = ManifoldModel(layers, embedding, x0, v0, config.holographic, config=config, store_full_sequence=store_full)
        
        # ── 7. Readout plugin ─────────────────────────────────────────────────
        readout_builder = ReadoutBuilder(config, dim_total, topology)
        readout_plugin = readout_builder.build()
        readout_plugin.register_hooks(model.hooks)
        model.add_module('readout_plugin', readout_plugin)
        
        # ── 8. Optional Plugins ───────────────────────────────────────────────
        # Pooling plugin
        pooling_builder = PoolingBuilder(config, topology)
        pooling_plugin = pooling_builder.build()
        if pooling_plugin:
            pooling_plugin.register_hooks(model.hooks)
            model.add_module('pooling_plugin', pooling_plugin)
        
        # Checkpointing plugin
        ckpt_builder = CheckpointingBuilder(config)
        ckpt_plugin = ckpt_builder.build()
        if ckpt_plugin:
            ckpt_plugin.register_hooks(model.hooks)
            model.add_module('checkpointing_plugin', ckpt_plugin)
        
        # Adjoint plugin
        adjoint_builder = AdjointBuilder(config)
        adjoint_plugin = adjoint_builder.build()
        if adjoint_plugin:
            adjoint_plugin.register_hooks(model.hooks)
            model.add_module('adjoint_plugin', adjoint_plugin)
        
        # Lensing plugin
        lensing_builder = LensingBuilder(config)
        lensing_plugin = lensing_builder.build()
        if lensing_plugin:
            lensing_plugin.register_hooks(model.hooks)
            model.add_module('lensing_plugin', lensing_plugin)
        
        return model

    @staticmethod
    def from_pretrained(save_directory: str) -> ManifoldModel:
        """
        Loads a ManifoldModel from a directory.
        Expects config.json and pytorch_model.bin.
        """
        config_path = os.path.join(save_directory, "config.json")
        model_path = os.path.join(save_directory, "pytorch_model.bin")

        if not os.path.exists(config_path):
            raise ConfigurationError(f"Config file not found in {save_directory}")
        if not os.path.exists(model_path):
            raise ConfigurationError(f"Model weights not found in {save_directory}")

        # 1. Load Config
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Reconstruct ManifoldConfig
        config = from_dict(ManifoldConfig, config_dict)

        # 2. Create Model Structure
        model = ModelFactory.create(config=config)

        # 3. Load Weights
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {save_directory}")
        return model
