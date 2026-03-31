import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from ..interfaces.geometry import Geometry
from ..interfaces.integrator import Integrator
from ..interfaces.physics import PhysicsEngine
from ..config.schema import PhysicsConfig
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
from .components.mixer import FlowMixer
from .plugins import LAYER_PLUGIN_REGISTRY
# Import plugins to trigger registration
from .plugins import dynamic_time, fractal
import logging

logger = logging.getLogger(__name__)


class ManifoldLayer(nn.Module):
    """
    GFN Manifold Layer with full feature parity with V4.

    Configuration via `PhysicsConfig`:
      topology.type             — TOPOLOGY_TORUS | TOPOLOGY_EUCLIDEAN
      stability.base_dt         — base time step
      stability.enable_trace_normalization — activates Riemannian norm
      dynamics.type (or kwargs)  — 'direct' | 'residual' | 'mix' | 'gated' | 'stochastic'
      active_inference.dynamic_time.enabled  — adaptive gating per head
      active_inference.dynamic_time.type     — 'riemannian' | 'thermo'
      fractal.enabled           — tunneling for high curvature
      fractal.threshold / alpha — fractal parameters
    """

    def __init__(
        self,
        integrator: Integrator,
        mixer: nn.Module,
        config: Optional[PhysicsConfig] = None,
        heads: int = 4,
        head_dim: Optional[int] = None,
        dynamics_type: str = 'direct',
        layer_idx: int = 0,
        total_depth: int = 6,
    ):
        super().__init__()
        self.integrator = integrator
        self.mixer = mixer
        self.config = config or PhysicsConfig()
        self.heads = heads
        self.layer_idx = layer_idx
        self.total_depth = total_depth

        # ── Topología ─────────────────────────────────────────────────────────
        self.topology = self.config.topology.type.lower()
        self.geometry_scope = getattr(self.config.topology, 'geometry_scope', 'local')

        # ── Head dim inferido del integrador/mixer si no se especifica ────────
        if head_dim is None:
            full_dim = getattr(mixer, 'dim', 64)
            self.head_dim = full_dim // heads
        else:
            self.head_dim = head_dim

        # ── Normalización geométrica ──────────────────────────────────────────
        from ..physics.normalization import ManifoldNormalizationRegistry
        use_norm = self.config.stability.enable_trace_normalization
        dim_total = self.heads * self.head_dim  # dimensión total para tensores aplanados
        
        # Extraer geometría del integrador para normalización metric-aware
        geometry = getattr(self.integrator.physics_engine, 'geometry', None)
        
        self.norm_x = ManifoldNormalizationRegistry.get_for_topology(
            self.topology, dim_total, is_velocity=False, geometry=geometry
        ) if use_norm else ManifoldNormalizationRegistry.get('identity')
        self.norm_v = ManifoldNormalizationRegistry.get_for_topology(
            self.topology, dim_total, is_velocity=True, geometry=geometry
        ) if use_norm else ManifoldNormalizationRegistry.get('identity')

        # ── Dynamics routing ──────────────────────────────────────────────────
        # Los dynamics se aplican sobre tensores aplanados [B, H*HD]
        dyn_type_cfg = getattr(self.config, 'dynamics', None)
        resolved_dyn_type = (
            dyn_type_cfg.type if dyn_type_cfg and dyn_type_cfg.type != 'direct'
            else dynamics_type
        )
        from ..physics.dynamics import get_dynamics
        self.dynamics_x = get_dynamics(
            resolved_dyn_type, dim_total, self.norm_x, topology=self.topology
        )
        self.dynamics_v = get_dynamics(
            resolved_dyn_type, dim_total, self.norm_v, topology=TOPOLOGY_EUCLIDEAN
        )
        self.dynamics_type = resolved_dyn_type

        # ── Mixer (obligatorio, NO es plugin) ──────────────────────────────────
        self.mixer_dim = self.heads * self.head_dim
        self.mixer = FlowMixer(self.mixer_dim, topology=self.topology)

        # ── Fractal Sub-Manifold (optional, now handled by fractal plugin) ─────
        # Keep config reference for backward compatibility
        self.fractal_enabled = getattr(self.config.fractal, 'enabled', False)

        # ── Plugin System ──────────────────────────────────────────────────────
        self.plugins = nn.ModuleDict()
        self._init_plugins()

    def _init_plugins(self) -> None:
        """Initialize plugins from registry based on config."""
        # Plugins already imported at module level to trigger registration
        # Create enabled plugins
        for plugin_name in LAYER_PLUGIN_REGISTRY.list_plugins():
            plugin = LAYER_PLUGIN_REGISTRY.create_plugin(
                plugin_name, self, self.config
            )
            if plugin is not None:
                self.plugins[plugin_name] = plugin
                plugin.setup()
                logger.debug(f"Layer {self.layer_idx}: Enabled plugin '{plugin_name}'")

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        force: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x, v: [B, S, H, D] (sequence with heads)
                  or [B, H, D]  (batch without sequence)
            force: [B, S, D] or [B, D] — external force

        Returns:
            (x_next, v_next) — same shape as input
        """
        original_shape = x.shape

        # Validate force shape compatibility
        if force is not None:
            if x.dim() == 4:
                B, S, H, D = x.shape
                if not ((force.dim() == 3 and force.shape == (B, S, self.heads * self.head_dim)) or
                        (force.dim() == 4 and force.shape == (B, S, H, D)) or
                        (self.geometry_scope == 'global' and force.dim() == 3 and force.shape == (B, S, self.head_dim))):
                    raise ValueError(f"Force shape {force.shape} incompatible with x shape {x.shape} for 4D x")
            elif x.dim() == 3:
                B, H, D = x.shape
                if not ((force.dim() == 2 and force.shape == (B, self.heads * self.head_dim)) or
                        (force.dim() == 3 and force.shape == (B, H, D)) or
                        (self.geometry_scope == 'global' and force.dim() == 2 and force.shape == (B, self.head_dim))):
                    raise ValueError(f"Force shape {force.shape} incompatible with x shape {x.shape} for 3D x")

        # 1. Reshape: homogenize to [B_eff, H, D]
        if x.dim() == 4:
            B, S = x.shape[:2]
            x_3d = x.reshape(B * S, self.heads, self.head_dim)
            v_3d = v.reshape(B * S, self.heads, self.head_dim)
            if force is not None:
                if force.dim() == 4:
                    f_3d = force.reshape(B * S, self.heads, self.head_dim)
                elif force.dim() == 3:
                    # force=[B, S, D] -> Expand to [B*S, H, D] if scope is global
                    f_3d = force.reshape(B * S, 1, -1).expand(-1, self.heads, -1)
                else:
                    f_3d = None
            else:
                f_3d = None
        elif x.dim() == 3:
            x_3d = x  # already is [B, H, D]
            v_3d = v
            if force is not None:
                if force.dim() == 2:
                    if self.geometry_scope == 'global':
                        # Each head sees full force
                        f_3d = force.unsqueeze(1).expand(-1, self.heads, -1)
                    else:
                        # Partition force [B, H*HD] -> [B, H, HD]
                        f_3d = force.reshape(x_3d.shape[0], self.heads, self.head_dim)
                elif force.dim() == 3:
                    f_3d = force
                else:
                    f_3d = None
            else:
                f_3d = None
        else:
            raise ValueError(f"ManifoldLayer: unsupported x shape: {x.shape}")

        B_eff = x_3d.shape[0]

        # 2. Plugin: Pre-integrate hooks (e.g., dynamic_time)
        dt_base = getattr(self.config.stability, 'base_dt', 0.1)
        dt_eff = dt_base
        for plugin in self.plugins.values():
            x_3d, v_3d, dt_eff = plugin.pre_integrate(x_3d, v_3d, dt_eff, f_3d)

        # 3. Integration step (vectorized over heads [B, H, D])
        x_prev, v_prev = x_3d, v_3d
        res = self.integrator.step(x_3d, v_3d, force=f_3d, dt=dt_eff)
        x_stepped, v_stepped = res["x"], res["v"]

        # 4. Plugin: Post-integrate hooks
        for plugin in self.plugins.values():
            x_stepped, v_stepped = plugin.post_integrate(
                x_stepped, v_stepped, x_prev, v_prev
            )

        # 5. Head mixing (required, not a plugin)
        x_mix, v_mix = self.mixer(x_stepped, v_stepped)

        # 6. Dynamics routing (applies mixing proposal)
        # x_mix can be [B, D] (partition) or [B, H, D] (ensemble)
        if x_mix.dim() == 2:
            # Partition mode: apply dynamics in flat space and redistribute
            x_ref_h = x_3d.reshape(B_eff, -1)
            v_ref_h = v_3d.reshape(B_eff, -1)
            x_next_flat = self.dynamics_x(x_ref_h, x_mix, context_x=x_ref_h)
            v_next_flat = self.dynamics_v(v_ref_h, v_mix, context_x=v_ref_h)
            x_next = x_next_flat.view(B_eff, self.heads, self.head_dim)
            v_next = v_next_flat.view(B_eff, self.heads, self.head_dim)
        else:
            # Ensemble mode: apply per head
            x_next = self.dynamics_x(x_3d.reshape(B_eff, -1),
                                     x_mix.reshape(B_eff, -1),
                                     context_x=x_3d.reshape(B_eff, -1)).view(B_eff, self.heads, self.head_dim)
            v_next = self.dynamics_v(v_3d.reshape(B_eff, -1),
                                     v_mix.reshape(B_eff, -1),
                                     context_x=v_3d.reshape(B_eff, -1)).view(B_eff, self.heads, self.head_dim)

        # Apply topology boundary wrapping to maintain manifold constraints
        x_next = self.integrator._resolve_topology(x_next)

        # 6. Plugin: Finalize hooks (e.g., fractal step)
        for plugin in self.plugins.values():
            x_next, v_next = plugin.finalize(x_next, v_next)

        # 7. Restore original shape
        if len(original_shape) == 4:
            B, S = original_shape[:2]
            x_next = x_next.view(B, S, self.heads, self.head_dim)
            v_next = v_next.view(B, S, self.heads, self.head_dim)

        return x_next, v_next

    # ── Helpers ───────────────────────────────────────────────────────────────

    # Removed _apply_dynamics_x/v as they were using stateful _last_x
    # Removed _fractal_step as it's now handled by fractal plugin

    def debug_state(self, x: torch.Tensor, v: torch.Tensor, label: str = "") -> None:
        """Utility for numerical health monitoring of layer state."""
        with torch.no_grad():
            x_mag = x.abs().mean().item()
            v_mag = v.abs().mean().item()
            has_nan = torch.isnan(x).any() or torch.isnan(v).any()
            logger.debug(f"Layer {self.layer_idx} ({label}): x_avg={x_mag:.4f}, v_avg={v_mag:.4f}, NaN={has_nan}")
            if has_nan:
                logger.warning(f"NaN detected in Layer {self.layer_idx}")


# Compatibility alias
MLayer = ManifoldLayer
