"""
Drone Detection — GFN V5 Model: Patch-based Multimodal Architecture
====================================================================
Fix estructural del problema de colapso espacial.

En vez de comprimir la imagen entera en 1 token:
   pixels [B, 3*H*W] → [B, 1, D]   ← pierde info espacial

Usamos PatchImageProjector que divide en una grilla 4×4:
   pixels [B, 3*H*W] → [B, 16, D]  ← 16 posiciones espaciales distintas

El GFN recibe 16 tokens, cada uno con info de una región del frame.
Esto permite al manifold distinguir "dron arriba-izquierda" vs "abajo-derecha".
"""

import torch
import torch.nn as nn
import math
import gfn

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PHYSICS_CONFIG = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',
        'coord_dim': 16,
        'impulse_scale': 80.0,
    },
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.05},
        'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.8},
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
    'topology': {
        'type': 'torus',
        'riemannian_type': 'low_rank',
    },
    'stability': {
        'enable_trace_normalization': False,
        'base_dt': 0.1,             # Reduced from 0.2 for finer AI control
        'velocity_saturation': 3.0, # Tighter bound to prevent chaotic wrapping
        'friction': 0.2,            # Base friction increased
        'toroidal_curvature_scale': 0.01,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PATCH IMAGE PROJECTOR — Fix de colapso espacial
# ══════════════════════════════════════════════════════════════════════════════

class PatchImageProjector(nn.Module):
    """
    Divide la imagen en patches y proyecta cada uno a espacio de fuerza del manifold.

    Con img_size=64 y patch_size=16:
      imagen [B, 3*64*64] → grilla 4×4 → 16 patches de [3*16*16=768] cada uno
      cada patch → [B, 1, D] via MLP independiente
      total → [B, 16, D]  ← el GFN ve 16 tokens con info espacial

    Esto permite al manifold aprender a asociar posiciones de tokens
    con posiciones en el frame, resolviendo el problema de localización.
    """
    def __init__(self, img_size: int, patch_size: int, dim: int,
                 impulse_scale: float = 80.0):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) debe ser divisible por patch_size ({patch_size})"

        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size          # ej: 64//16 = 4
        self.n_patches  = self.grid_size ** 2              # ej: 4×4 = 16
        self.patch_dim  = 3 * patch_size * patch_size      # ej: 3×16×16 = 768

        # MLP compartida para todos los patches (eficiente)
        self.proj = nn.Sequential(
            nn.Linear(self.patch_dim, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        # Positional embedding: codifica posición espacial de cada patch
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)

        # impulse_scale bajo: la normalización L2 en forward hace que esta sea
        # la magnitud real del impulso, independiente del img_size.
        self.impulse_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3*H*W]  → [B, n_patches, dim]
        """
        B = x.shape[0]
        H = W = self.img_size
        P = self.patch_size
        G = self.grid_size

        # Reshape a imagen 3D
        img = x.view(B, 3, H, W)                         # [B, 3, H, W]

        # Extraer patches: [B, 3, H, W] → [B, n_patches, patch_dim]
        # Usando unfold para hacerlo eficientemente sin copias
        img = img.unfold(2, P, P).unfold(3, P, P)         # [B, 3, G, G, P, P]
        img = img.contiguous().view(B, 3, G*G, P*P)       # [B, 3, n_patches, P²]
        img = img.permute(0, 2, 1, 3)                     # [B, n_patches, 3, P²]
        patches = img.reshape(B, self.n_patches, -1)       # [B, n_patches, 3*P*P]

        # Proyectar cada patch a dim
        forces = self.proj(patches)                        # [B, n_patches, dim]

        # L2-normalizar la proyección pura (SIN pos_embed todavía).
        # Esto desacopla la magnitud del proyector del img_size:
        # un img=128 y un img=32 producirán la misma escala de fuerza.
        norms  = forces.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        forces = forces / norms                            # [B, n_patches, dim] — unitario

        # DESPUÉS de normalizar: sumar el bias posicional (pequeño, ~0.02 escala)
        # y escalar por impulse_scale (magnitud real del impulso).
        forces = forces + self.pos_embed                   # añade info espacial
        return forces * self.impulse_scale                 # [B, n_patches, dim]


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION HEAD — Lee x_final y predice [obj, cx, cy, w, h]
# ══════════════════════════════════════════════════════════════════════════════

class DetectionHead(nn.Module):
    """
    Lee la posición final del manifold (x_final) y predice [obj, cx, cy, w, h].
    Lazy init — dimensiones inferidas del tensor x_final en el primer forward.
    """
    def __init__(self):
        super().__init__()
        self.head = None

    def forward(self, x_final: torch.Tensor) -> torch.Tensor:
        """
        x_final: [B, H, D] o [B, D]
        Returns: [B, 5] en [-pi, pi]
        """
        x_flat = x_final.flatten(1)                       # [B, H*D]
        x_feat = torch.cat([
            torch.sin(x_flat),
            torch.cos(x_flat),
        ], dim=-1)                                          # [B, 2*H*D]

        if self.head is None:
            feat_dim = x_feat.shape[-1]
            mid_dim  = max(feat_dim // 4, 64)
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, 5),
            ).to(x_final.device)

        pred = self.head(x_feat)                           # [B, 5]
        return torch.tanh(pred) * math.pi                  # → [-pi, pi]


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_detector(
    img_size:   int   = 64,    # subido de 32 a 64 para mejor resolución
    patch_size: int   = 32,    # patches de 16×16 → 4×4 grilla = 16 tokens
    dim:        int   = 32,
    depth:      int   = 1,
    heads:      int   = 2,
    rank:       int   = 32,
):
    """
    Construye el sistema de detección GFN con patch projection.

    Returns: (projector, gfn_model, detection_head)
    
    Pipeline:
      forces = projector(img)          # [B, 16, D] — 16 tokens espaciales
      _, (x_f, _), _ = gfn(force_manual=forces)
      pred = det_head(x_f)             # [B, 5] — [obj, cx, cy, w, h]
    """
    projector = PatchImageProjector(
        img_size=img_size,
        patch_size=patch_size,
        dim=dim,
    )

    manifold = gfn.create(
        vocab_size=2,
        dim=dim,
        depth=depth,
        heads=heads,
        rank=rank,
        physics=PHYSICS_CONFIG,
        holographic=True,
        trajectory_mode='partition',
        coupler_mode='mean_field',
        initial_spread=0.01,
        integrator='yoshida',
    )

    det_head = DetectionHead()

    return projector, manifold, det_head
