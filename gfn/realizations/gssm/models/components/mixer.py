"""
gfn/models/components/mixer.py — GFN V5
Portado y unificado desde: gfn_old/nn/layers/flow/mixer.py

FlowMixer: Único mezclador unificado para ManifoldLayer.

Modos disponibles:
  'low_rank'  — [B, H, D] → [B, D]  via proyección Low-Rank (partición)
  'attention' — [B, H, D] → [B, D]  via Geodesic Attention (partición)
  'ensemble'  — [B, H, D] → [B, H, D] via consenso ponderado (preserva trayectorias)
  'geodesic'  — Alias de 'attention' con temperatura adaptativa

Topología toro: posiciones se proyectan vía [sin(x), cos(x)] para circular averaging.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN


class FlowMixer(nn.Module):
    """
    Mezclador unificado de estado geodésico para ManifoldLayer V5.

    En modo 'low_rank' y 'attention' colapsa cabezas a estado único [B, D].
    En modo 'ensemble' preserva la estructura de trayectorias [B, H, D].
    """

    def __init__(self, dim: int, rank: int = 16, heads: int = 4,
                 topology: str = TOPOLOGY_EUCLIDEAN, mode: str = 'low_rank',
                 use_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.heads = heads
        self.topology = str(topology).lower().strip()
        self.mode = str(mode).lower().strip()
        self.use_norm = use_norm
        self.head_dim = dim // heads if heads > 0 else dim

        if self.mode in ('low_rank', 'default', 'geodesic', 'attention'):
            self._build_partition_mixer()
        elif self.mode == 'ensemble':
            self._build_ensemble_coupler()
        else:
            # Fallback seguro
            self.mode = 'low_rank'
            self._build_partition_mixer()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build_partition_mixer(self):
        """Mixer de partición: colapsa [B, H, D] → [B, D]."""
        # Torus: proyectar via [sin(x), cos(x), tanh(v/10)] — 3×dim de entrada
        # Euclidean: proyectar vía x directamente — 1×dim de entrada
        mixer_in_x_dim = (3 if self.topology == TOPOLOGY_TORUS else 1) * self.dim
        self.out_proj_x = nn.Linear(mixer_in_x_dim, self.dim)
        self.out_proj_v = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.out_proj_x.weight)
        nn.init.zeros_(self.out_proj_x.bias)
        nn.init.xavier_uniform_(self.out_proj_v.weight)
        nn.init.zeros_(self.out_proj_v.bias)
        
        # Consistencia física: la mezcla de velocidad no debe sphericalizarse vía RMSNorm
        # ya que destruiría la información de magnitud acumulada (momentum).
        # Usamos Identity y dejamos que el regulador dinámico de la capa maneje el clamp.
        self.mixed_norm_v = nn.Identity()

    def _build_ensemble_coupler(self):
        """Ensemble coupler: mantiene [B, H, D] con consenso ponderado."""
        self.ensemble_attn = nn.Parameter(torch.ones(self.heads) / self.heads)
        self.coupling_proj = nn.Linear(self.head_dim, self.head_dim)
        if self.use_norm:
            self.mixed_norm_v = nn.RMSNorm(self.head_dim)
        else:
            self.mixed_norm_v = nn.Identity()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, H, D] — estado de posición por cabeza
            v: [B, H, D] — estado de velocidad por cabeza (opcional, usa x si None)
            history: ignorado (compat. legacy JacobiAttention)

        Returns:
            (x_out, v_out) — formas según modo:
              low_rank/attention: [B, D]
              ensemble:           [B, H, D]
        """
        if v is None:
            v = x

        # Caso especial: cabeza única
        if self.heads == 1 or (x.dim() == 3 and x.shape[1] == 1):
            return x.squeeze(1) if x.dim() == 3 else x, \
                   v.squeeze(1) if v.dim() == 3 else v

        if self.mode == 'ensemble':
            return self._forward_ensemble(x, v)
        else:
            return self._forward_partition(x, v)

    def _forward_partition(self, x: torch.Tensor, v: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, H, head_dim] → [B, D]"""
        B, H, HD = x.shape
        
        if self.topology == TOPOLOGY_TORUS:
            # Multi-head circular averaging: atan2(avg(sin), avg(cos))
            # We use einsum to allow learned head weighting if needed, 
            # but for FlowMixer we just average or let the linear layer handle it per channel.
            sin_x = torch.sin(x) # [B, H, HD]
            cos_x = torch.cos(x) # [B, H, HD]
            v_scaled = torch.tanh(v / 10.0) # [B, H, HD]
            
            # Flatten to [B, 3*H*HD] for out_proj_x compatibility if it was built that way,
            # but we should ensure out_proj_x sees head-separated features.
            # Current out_proj_x expects (3 * dim) where dim = H*HD
            x_flat_cat = torch.cat([sin_x.reshape(B, -1), cos_x.reshape(B, -1), v_scaled.reshape(B, -1)], dim=-1)
            x_agg = self.out_proj_x(x_flat_cat)
            
            # Final projection to manifold
            x_agg = torch.atan2(torch.sin(x_agg), torch.cos(x_agg))
        else:
            x_flat = x.reshape(B, -1)
            x_agg = self.out_proj_x(x_flat)

        v_flat = v.reshape(B, -1)
        v_agg = self.mixed_norm_v(self.out_proj_v(v_flat))

        return x_agg, v_agg

    def _forward_ensemble(self, x: torch.Tensor, v: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, H, D] → consenso ponderado → [B, H, D]"""
        weights = torch.softmax(self.ensemble_attn, dim=0).view(1, -1, 1)
        
        if self.topology == TOPOLOGY_TORUS:
            # Geodesic consensus: atan2(sum(w*sin), sum(w*cos))
            sin_x = torch.sin(x)
            cos_x = torch.cos(x)
            x_center = torch.atan2((sin_x * weights).sum(dim=1, keepdim=True),
                                  (cos_x * weights).sum(dim=1, keepdim=True))
        else:
            x_center = (x * weights).sum(dim=1, keepdim=True)   # [B, 1, D]
            
        v_center = (v * weights).sum(dim=1, keepdim=True)   # [B, 1, D]

        # Converger cada trayectoria hacia el consenso (acoplamiento suave)
        if self.topology == TOPOLOGY_TORUS:
            diff_x = x_center - x
            # Wrapped difference for torus
            diff_x = torch.atan2(torch.sin(diff_x), torch.cos(diff_x))
            x_coupled = x + 0.1 * torch.tanh(self.coupling_proj(diff_x))
        else:
            x_coupled = x + 0.1 * torch.tanh(self.coupling_proj(x_center - x))
            
        v_coupled = v + 0.1 * torch.tanh(self.coupling_proj(v_center - v))

        return x_coupled, v_coupled


class GeodesicAttentionMixer(nn.Module):
    """
    Geodesic Attention Mixer — mezcla cabezas vía distancias Riemannianas.

    Modo attention: las cabezas más cercanas en el espacio geodésico
    tienen mayor peso en la mezcla.
    """

    def __init__(self, dim: int, heads: int, temperature: float = 1.0,
                 topology: str = TOPOLOGY_EUCLIDEAN):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.topology = str(topology).lower().strip()
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(self.head_dim, self.head_dim)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

    def _compute_distance(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Distancias pairwise [B, H, H]."""
        if self.topology == TOPOLOGY_TORUS:
            q_exp = q.unsqueeze(2)
            k_exp = k.unsqueeze(1)
            diff = q_exp - k_exp
            wrapped_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            return (wrapped_diff ** 2).sum(dim=-1)

        # Euclidean
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)
        k_sq = (k ** 2).sum(dim=-1).unsqueeze(1)
        dot = torch.bmm(q, k.transpose(1, 2))
        return torch.clamp(q_sq + k_sq - 2 * dot, min=0.0)

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, H, D] → [B, D]"""
        if v is None:
            v = x

        # Single head shortcut
        if x.shape[1] == 1:
            return x.squeeze(1), v.squeeze(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        val = self.v_proj(x)

        distances = self._compute_distance(q, k)
        attn_weights = torch.softmax(-distances / self.temperature, dim=-1)

        # Mezclar posiciones y velocidades
        if self.topology == TOPOLOGY_TORUS:
            # Geodesic Averaging: atan2(sum(w*sin), sum(w*cos))
            sin_x = torch.sin(x)
            cos_x = torch.cos(x)
            sum_sin = torch.bmm(attn_weights, sin_x)
            sum_cos = torch.bmm(attn_weights, cos_x)
            x_mixed = torch.atan2(sum_sin, sum_cos).flatten(1)
        else:
            x_mixed = torch.bmm(attn_weights, x).flatten(1)
            
        v_mixed = torch.bmm(attn_weights, v).flatten(1)

        return self.out_proj(x_mixed), self.out_proj(v_mixed)


# Alias hacia atrás
ManifoldMixer = FlowMixer
