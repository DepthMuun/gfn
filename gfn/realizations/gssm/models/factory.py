"""
gfn/models/factory.py — GFN V5
ModelFactory: construye modelos ManifoldModel completos desde ManifoldConfig.

Soporte para configuración vía:
  - ManifoldConfig directo (config=...)
  - Preset + overrides planos: gfn.create(preset_name='stable-torus', dim=64, ...)
  - Preset + dict de física: gfn.create(preset_name='...', physics={'stability': {'base_dt': 0.5}})
  - Dict de física puro sin preset: gfn.create(config=ManifoldConfig(physics=dict_to_physics_config({...})))
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
    PoolingBuilder, CheckpointingBuilder, AdjointBuilder
)

from ..config.serialization import from_dict


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory para construir modelos GFN V5.

    IMPORTANT: Geometry opera sobre tensores per-head [B, H, HD] donde HD = dim/heads.
    El factory pasa head_dim a GeometryFactory, no el dim total.

    Flujos de creación soportados:
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
    def create(
        config: Optional[ManifoldConfig] = None,
        preset_name: Optional[str] = None,
        physics: Optional[Union[Dict[str, Any], PhysicsConfig]] = None,
        **kwargs
    ) -> ManifoldModel:
        """
        Construye un ManifoldModel.

        Args:
            config:      ManifoldConfig completo. Si se provee, tiene prioridad.
            preset_name: (DEPRECADO) Nombre del preset de física.
            physics:     Dict anidado o PhysicsConfig para sobreescribir la física.
            **kwargs:    Overrides planos de ManifoldConfig/PhysicsConfig.
                         Soporta prefijos para llegar a niveles anidados:
                         - 'topology_type', 'base_dt', 'friction', 'integrator'
        """
        # ── 0. Resolver configuración base ───────────────────────────────────
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
            
            config = from_dict(ManifoldConfig, config)

        # Re-sync with extra kwargs (they take priority over dict values if provided twice)
        explicit_keys.update(kwargs.keys())

        if config is None:
            vsize = kwargs.pop('vocab_size', 100)
            # Inicializar con defaults profesionales
            config = ManifoldConfig(vocab_size=vsize)

        # ── 1. Aplicar Overrides de Física (Dict/Config) ─────────────────────
        if physics is not None:
            if isinstance(physics, dict):
                apply_physics_overrides(config.physics, physics)
            elif isinstance(physics, PhysicsConfig):
                config.physics = physics
            else:
                raise ConfigurationError(f"physics must be a dict or PhysicsConfig, got {type(physics)}")

        # ── 2. Normalizar Configuración (usando ConfigNormalizer) ─────────────
        remaining_kwargs, validation_errors = normalize_config(config, kwargs, explicit_keys)
        if validation_errors:
            logger.warning(f"Config validation warnings: {validation_errors}")
        
        # Los kwargs restantes no fueron mapeados (posiblemente argumentos desconocidos)
        if remaining_kwargs:
            logger.debug(f"Unmapped kwargs: {list(remaining_kwargs.keys())}")

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
        
        # ── 5. Estado inicial ─────────────────────────────────────────────────
        spread = getattr(config, 'initial_spread', 1e-3)
        x0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
        v0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
        
        # ── 6. Ensamblado del modelo ───────────────────────────────────────────
        model = ManifoldModel(layers, embedding, x0, v0, config.holographic, config=config)
        
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
