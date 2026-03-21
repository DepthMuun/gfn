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
import json
import os
import logging
from typing import Optional, Dict, Any, Union, Set

from .manifold import ManifoldModel
from .manifold_layer import ManifoldLayer
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

        # ── 2. Mapeo de Kwargs Planos y Dotted ──────────────────────────────
        for k, v in list(kwargs.items()):
            # A. Intento recursivo si hay puntos (e.g. 'physics.topology.type')
            if '.' in k:
                try:
                    ModelFactory._recursive_setattr(config, k, v)
                    kwargs.pop(k)
                    continue
                except (AttributeError, KeyError):
                    pass # Intentar con las otras reglas si falla

            # B. Buscar en el primer nivel de ManifoldConfig
            if hasattr(config, k):
                setattr(config, k, v)
                kwargs.pop(k)
                continue

            # C. Intento directo en sub-configs de física (e.g. 'base_dt' -> stability)
            found = False
            for sub_name in ['topology', 'stability', 'dynamics', 'active_inference', 'embedding', 'readout', 'mixture', 'fractal', 'hysteresis', 'singularities']:
                target = getattr(config.physics, sub_name, None)
                if target and hasattr(target, k):
                    setattr(target, k, v)
                    kwargs.pop(k)
                    found = True
                    break
            if found: continue

            # D. Intento por prefijo solo para sub-configs válidos (e.g. 'topology_type')
            if '_' in k:
                for prefix in ['topology', 'stability', 'dynamics', 'embedding', 'readout', 'mixture', 'fractal', 'hysteresis', 'singularities']:
                    if k.startswith(prefix + '_'):
                        real_k = k[len(prefix)+1:]
                        apply_physics_overrides(config.physics, {prefix: {real_k: v}})
                        kwargs.pop(k)
                        found = True
                        break
                if found: continue
                
                # Caso especial para active_inference (contiene '_')
                if k.startswith('active_inference_'):
                    real_k = k[len('active_inference_')+1:]
                    apply_physics_overrides(config.physics, {'active_inference': {real_k: v}})
                    kwargs.pop(k)
                    continue

        # ── 3. Sincronizar parámetros entre ManifoldConfig y PhysicsConfig ─────
        # Priorizamos ManifoldConfig si el valor fue provisto explícitamente en kwargs
        # de lo contrario sincronizamos en ambas direcciones.

        # 1. Integrator
        if 'integrator' in explicit_keys:
            config.physics.stability.integrator_type = config.integrator
        else:
            config.integrator = config.physics.stability.integrator_type

        # 2. Impulse Scale
        if 'impulse_scale' in explicit_keys:
            config.physics.embedding.impulse_scale = config.impulse_scale
        else:
            config.impulse_scale = config.physics.embedding.impulse_scale

        # 3. Rank
        if 'rank' in explicit_keys:
            config.physics.topology.riemannian_rank = config.rank
        else:
            config.rank = config.physics.topology.riemannian_rank

        # 4. Dynamics Type
        if 'dynamics_type' in explicit_keys:
            config.physics.dynamics.type = config.dynamics_type
        else:
            config.dynamics_type = config.physics.dynamics.type

        # 5. Trajectory Mode
        if 'trajectory_mode' in explicit_keys:
            config.physics.trajectory_mode = config.trajectory_mode
        else:
            config.trajectory_mode = config.physics.trajectory_mode

        # 6. Coupler Mode
        if 'coupler_mode' in explicit_keys:
            config.physics.mixture.coupler_mode = config.coupler_mode
        else:
            config.coupler_mode = config.physics.mixture.coupler_mode

        # 7. Holographic
        if 'holographic' in explicit_keys:
            config.physics.active_inference.holographic_geometry = config.holographic
        else:
            # Sincronizar bidireccionalmente: si cualquiera es True, activar.
            # Pero prioritizar config.holographic si se pasó un objeto config pre-armado.
            config.holographic = config.holographic or config.physics.active_inference.holographic_geometry
            config.physics.active_inference.holographic_geometry = config.holographic

        topology_cfg = config.physics.topology
        geometry_scope = getattr(topology_cfg, 'geometry_scope', 'local')
        
        if geometry_scope == 'global':
            # GDG Mode: Each head has the full dim D. Total state is H * D.
            head_dim = config.dim
        else:
            # Local Mode: Heads partition the dim D. Total state is D.
            head_dim = config.dim // config.heads
            
        dim_total = config.heads * head_dim
        topology      = topology_cfg.type
        dynamics_type = config.dynamics_type
        mixer_type    = getattr(config, 'mixer_type', 'low_rank')

        # ── 4. Token Embedding ────────────────────────────────────────────────
        embedding = FunctionalEmbedding(
            vocab_size=config.vocab_size,
            emb_dim=config.dim,
            coord_dim=config.physics.embedding.coord_dim,
            mode=config.physics.embedding.mode,
            impulse_scale=config.impulse_scale,
        )

        # ── 5. Layers ─────────────────────────────────────────────────────────
        layers = nn.ModuleList()
        # Deep copy the base physics config to ensure modifications in one layer don't affect others
        # but the config is currently a static object, so we just use it.
        # The key is to create NEW instances of PhysicsEngine and Integrator.
        for layer_idx in range(config.depth):
            # Each layer MUST have its own Geometry, PhysicsEngine and Integrator instances
            # to prevent stateful leakage (e.g. Hysteresis memory bleeding between layers)
            geometry = GeometryFactory.create_with_dim(head_dim, config.rank, config.heads, config.physics)
            physics_engine = ManifoldPhysicsEngine(geometry, config.physics, dim=head_dim, heads=config.heads)
            integrator = IntegratorFactory.create(physics_engine, config.physics)

            mixer: nn.Module
            if mixer_type == 'attention':
                mixer = GeodesicAttentionMixer(dim_total, config.heads, topology=topology)
            else:
                mixer = FlowMixer(
                    dim=dim_total,
                    rank=config.rank,
                    heads=config.heads,
                    topology=topology,
                    mode=mixer_type,
                    use_norm=config.physics.stability.enable_trace_normalization,
                )

            layer = ManifoldLayer(
                integrator=integrator,
                mixer=mixer,
                config=config.physics,
                heads=config.heads,
                head_dim=head_dim,
                dynamics_type=dynamics_type,
                layer_idx=layer_idx,
                total_depth=config.depth,
            )
            layers.append(layer)

        # ── 6. Estado inicial ─────────────────────────────────────────────────
        spread = getattr(config, 'initial_spread', 1e-3)
        x0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
        v0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)

        # ── 7. Ensamblado del modelo ───────────────────────────────────────────
        model = ManifoldModel(layers, embedding, x0, v0, config.holographic, config=config)

        # ── 8. Readout plugin ─────────────────────────────────────────────────
        # P2.2 FIX: `holographic` controls geometry scope, NOT the readout type.
        # Readout type is independently determined by config.physics.readout.type.
        # This allows holographic geometry + implicit readout (e.g. for regression tasks).
        readout_type = config.physics.readout.type

        if readout_type == 'implicit':
            out_dim = getattr(config.physics.readout, 'out_dim', config.vocab_size)
            hidden_dim = getattr(config.physics.readout, 'hidden_dim', 128)
            readout = ImplicitReadout(
                dim_total, out_dim,
                hidden_dim=hidden_dim,
                topology_type=topology,
            )
        elif readout_type == 'identity':
            # Explicitly requested: return manifold state directly (latent readout)
            readout = IdentityReadout()
        elif readout_type == 'standard' and config.holographic:
            # Legacy behavior: holographic + standard → identity (backward compat)
            readout = IdentityReadout()
        else:
            readout = CategoricalReadout(dim_total, config.vocab_size, topology_type=topology)
            
        plugin = ReadoutPlugin(readout)
        plugin.register_hooks(model.hooks)
        model.add_module('readout_plugin', plugin)

        # ── 9. Pooling plugin (Optional) ──────────────────────────────────────
        pooling_type = getattr(config, 'pooling_type', None)
        if pooling_type:
            if pooling_type == 'hamiltonian':
                pool_mod = HamiltonianPooling(config.dim, topology_type=topology)
            elif pooling_type == 'hierarchical':
                # HierarchicalAggregator needs topology to pass down to HamiltonianPooling
                pool_mod = HierarchicalAggregator(config.dim, topology_type=topology)
            elif pooling_type == 'momentum':
                pool_mod = MomentumAggregator(config.dim, topology_type=topology)
            else:
                pool_mod = None
            
            if pool_mod:
                pool_plugin = PoolingPlugin(pool_mod)
                pool_plugin.register_hooks(model.hooks)
                model.add_module('pooling_plugin', pool_plugin)

        # ── 10. Checkpointing plugin ──────────────────────────────────────────
        ckpt_cfg = config.physics.checkpointing
        if ckpt_cfg.get('enabled', False):
            from ..models.components.checkpointing import CheckpointingPlugin
            ckpt_plugin = CheckpointingPlugin(chunk_size=ckpt_cfg.get('chunk_size', 32))
            ckpt_plugin.register_hooks(model.hooks)
            model.add_module('checkpointing_plugin', ckpt_plugin)

        # ── 11. Adjoint plugin ────────────────────────────────────────────────
        if getattr(config, 'adjoint_enabled', False):
            from ..models.components.adjoint import AdjointPlugin
            adj_plugin = AdjointPlugin(config)
            adj_plugin.register_hooks(model.hooks)
            model.add_module('adjoint_plugin', adj_plugin)

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
