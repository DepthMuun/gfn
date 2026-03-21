import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from ..interfaces.geometry import Geometry
from ..interfaces.integrator import Integrator
from ..interfaces.physics import PhysicsEngine
from ..config.schema import PhysicsConfig
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
import logging

logger = logging.getLogger(__name__)


class ManifoldLayer(nn.Module):
    """
    Capa de Manifold GFN V5 con feature parity completo respecto a V4.

    Configuración via `PhysicsConfig`:
      topology.type             — TOPOLOGY_TORUS | TOPOLOGY_EUCLIDEAN
      stability.base_dt         — step de tiempo base
      stability.enable_trace_normalization — activa norm Riemanniana
      dynamics.type (o kwargs)  — 'direct' | 'residual' | 'mix' | 'gated' | 'stochastic'
      active_inference.dynamic_time.enabled  — gating adaptativo por cabeza
      active_inference.dynamic_time.type     — 'riemannian' | 'thermo'
      fractal.enabled           — tunneling por curvatura alta
      fractal.threshold / alpha — parámetros del fractal
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

        # ── Time-step dinámico por cabeza ─────────────────────────────────────
        dt_cfg = self.config.active_inference.dynamic_time
        self.use_dynamic_time = dt_cfg.enabled
        self.dynamic_time_type = dt_cfg.type.lower()

        from ..physics.gating import RiemannianGating, ThermodynamicLayer
        if self.dynamic_time_type == 'thermo':
            self.gatings = nn.ModuleList([
                ThermodynamicLayer(self.head_dim) for _ in range(heads)
            ])
        else:
            self.gatings = nn.ModuleList([
                RiemannianGating(self.head_dim, topology=self.topology)
                for _ in range(heads)
            ])

        # dt por cabeza: parámetros trainables con softplus scaling
        self.base_dt = self.config.stability.base_dt
        scale_vals: List[torch.Tensor] = []
        for i in range(heads):
            target_dt = self.base_dt / 0.9
            val_init = torch.log(torch.exp(torch.tensor(target_dt)) - 1.0)
            self.dt_increment = 0.05
            scale_vals.append(val_init + i * self.dt_increment)
        self.dt_params = nn.Parameter(torch.stack(scale_vals))

        # ── Fractal Sub-Manifold (opcional) ───────────────────────────────────
        fract_cfg = self.config.fractal
        self.fractal_enabled = fract_cfg.enabled
        self.fractal_threshold = fract_cfg.threshold
        self.fractal_alpha = fract_cfg.alpha
        self.micro_manifold: Optional[nn.Module] = None  # Explicit type

        self.fractal_slope = 1.0
        self.dt_increment = 0.05

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        force: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x, v: [B, S, H, D] (secuencia con cabezas)
                  o [B, H, D]  (batch sin secuencia)
            force: [B, S, D] o [B, D] — fuerza externa

        Returns:
            (x_next, v_next) — misma forma que entrada
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

        # 1. Reshape: homogeneizar a [B_eff, H, D]
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
            x_3d = x  # ya es [B, H, D]
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
            raise ValueError(f"ManifoldLayer: forma de x no soportada: {x.shape}")

        B_eff = x_3d.shape[0]

        # 2. Gating del dt por cabeza
        # Pre-convert to avoid repeated softplus if dynamic time is disabled
        dt_base = torch.nn.functional.softplus(self.dt_params).view(1, self.heads, 1)
        dt_base = torch.clamp(dt_base, self.config.stability.dt_min, self.config.stability.dt_max)

        if self.use_dynamic_time:
            if self.dynamic_time_type == 'thermo':
                gates_list = [
                    self.gatings[i](x_3d[:, i], v_3d[:, i])
                    for i in range(self.heads)
                ]
            else:
                gates_list = [
                    self.gatings[i](x_3d[:, i])
                    for i in range(self.heads)
                ]
            gates = torch.stack(gates_list, dim=1)   # [B_eff, H, 1]
            dt_eff = dt_base * gates
        else:
            dt_eff = dt_base  # escalar broadcast sobre [B, H, 1]

        # 3. Paso de integración (vectorizado sobre cabezas [B, H, D])
        x_stepped, v_stepped = x_3d, v_3d
        
        # Validar si el integrador soporta dt como tensor [B, H, 1]
        res = self.integrator.step(x_3d, v_3d, force=f_3d, dt=dt_eff)
        x_stepped = res["x"]
        v_stepped = res["v"]

        # 4. Mixing de cabezas
        x_ref_2d = x_3d.reshape(B_eff, -1)
        v_ref_2d = v_3d.reshape(B_eff, -1)
        # No longer storing _last_x as attribute to avoid gradient corruption in shared layers
        x_mix, v_mix = self.mixer(x_stepped, v_stepped)

        # 5. Dynamics routing (actualiza estado con la propuesta del mixing)
        # x_mix puede ser [B, D] (partición) o [B, H, D] (ensemble)
        if x_mix.dim() == 2:
            # Modo partición: aplicar dynamics en espacio aplanado y redistribuir
            x_ref_h = x_3d.reshape(B_eff, -1)
            v_ref_h = v_3d.reshape(B_eff, -1)
            x_next_flat = self.dynamics_x(x_ref_h, x_mix, context_x=x_ref_h)
            v_next_flat = self.dynamics_v(v_ref_h, v_mix, context_x=x_ref_h)
            x_next = x_next_flat.view(B_eff, self.heads, self.head_dim)
            v_next = v_next_flat.view(B_eff, self.heads, self.head_dim)
        else:
            # Modo ensemble: aplicar por cabeza
            x_next = self.dynamics_x(x_3d.reshape(B_eff, -1),
                                     x_mix.reshape(B_eff, -1),
                                     context_x=x_3d.reshape(B_eff, -1)).view(B_eff, self.heads, self.head_dim)
            v_next = self.dynamics_v(v_3d.reshape(B_eff, -1),
                                     v_mix.reshape(B_eff, -1),
                                     context_x=x_3d.reshape(B_eff, -1)).view(B_eff, self.heads, self.head_dim)

        # Apply topology boundary wrapping to maintain manifold constraints
        x_next = self.integrator._resolve_topology(x_next)

        # 6. Fractal step opcional
        if self.fractal_enabled and self.micro_manifold is not None:
            x_next, v_next = self._fractal_step(x_next, v_next, f_3d, B_eff)

        # 7. Restaurar forma original
        if len(original_shape) == 4:
            B, S = original_shape[:2]
            x_next = x_next.view(B, S, self.heads, self.head_dim)
            v_next = v_next.view(B, S, self.heads, self.head_dim)

        return x_next, v_next

    # ── Helpers ───────────────────────────────────────────────────────────────

    # Removed _apply_dynamics_x/v as they were using stateful _last_x

    def _fractal_step(self, x: torch.Tensor, v: torch.Tensor,
                      force, B_eff: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica el micro-manifold si la curvatura supera el threshold."""
        # Estimación simple de curvatura: norma del estado de velocidad
        curvature_est = v.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        # Simple proxy for curvature: average norm of velocity vectors
        tunnel_gate = torch.sigmoid((curvature_est - self.fractal_threshold) * self.fractal_slope)

        x_f, v_f = self.micro_manifold(x, v, force=force)
        x_out = x + tunnel_gate * (x_f - x) * self.fractal_alpha
        v_out = v + tunnel_gate * (v_f - v) * self.fractal_alpha
        return x_out, v_out

    def debug_state(self, x: torch.Tensor, v: torch.Tensor, label: str = "") -> None:
        """Utilidad de monitoreo de salud numérica del estado de la capa."""
        with torch.no_grad():
            x_mag = x.abs().mean().item()
            v_mag = v.abs().mean().item()
            has_nan = torch.isnan(x).any() or torch.isnan(v).any()
            logger.debug(f"Layer {self.layer_idx} ({label}): x_avg={x_mag:.4f}, v_avg={v_mag:.4f}, NaN={has_nan}")
            if has_nan:
                logger.warning(f"NaN detected in Layer {self.layer_idx}")


# Alias de compatibilidad
MLayer = ManifoldLayer
