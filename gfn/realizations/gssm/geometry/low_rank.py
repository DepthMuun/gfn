"""
LowRankRiemannianGeometry — GFN V5
Computes Christoffel symbols via a low-rank decomposition.
Migrated from gfn/geo/riemannian/low_rank_geometry.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any

from ..constants import (
    EPS, MAX_VELOCITY, TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN,
    DEFAULT_FRICTION, CURVATURE_CLAMP, GATE_BIAS_OPEN
)
from ..config.schema import PhysicsConfig
from ..geometry.base import BaseGeometry
from ..registry import register_geometry
from ..cuda.ops import CUDA_AVAILABLE, low_rank_christoffel_fwd, low_rank_christoffel_bwd

class LowRankChristoffelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, U, W, clamp_val, enable_trace_norm, is_paper_version=False):
        v_c = v.contiguous()
        U_c = U.contiguous()
        W_c = W.contiguous()
        
        gamma = low_rank_christoffel_fwd(v_c, U_c, W_c, float(clamp_val), enable_trace_norm, is_paper_version)
        ctx.save_for_backward(v_c, U_c, W_c, gamma)
        ctx.clamp_val = float(clamp_val)
        ctx.enable_trace_norm = enable_trace_norm
        ctx.is_paper_version = is_paper_version
        return gamma

    @staticmethod
    def backward(ctx, grad_gamma):
        v_c, U_c, W_c, gamma_out = ctx.saved_tensors
        if grad_gamma is None:
            return None, None, None, None, None, None
            
        grad_gamma_c = grad_gamma.contiguous()
        d_v, d_U, d_W = low_rank_christoffel_bwd(
            grad_gamma_c, v_c, U_c, W_c, gamma_out, 
            ctx.clamp_val, ctx.enable_trace_norm, ctx.is_paper_version
        )
        return d_v, d_U, d_W, None, None, None


# Use unified FrictionGate from physics.components (no duplication)
from ..physics.components.friction import FrictionGate


@register_geometry('low_rank')
class LowRankRiemannianGeometry(BaseGeometry):
    r"""
    Low-rank Christoffel symbol decomposition.

    Γ^k_ij ≈ Σ_r W_{rk} * (U_ir * U_jr)
    This is an approximation — symmetry is preserved but Bianchi identities are not guaranteed.
    Chosen for computational efficiency.

    Args:
        dim: Manifold dimension.
        rank: Rank of the decomposition.
        num_heads: Number of parallel heads.
        config: PhysicsConfig instance.
    """

    def __init__(self, dim: int, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self.dim = dim
        self.rank = rank
        self.num_heads = num_heads

        topo = self.config.topology.type.lower()
        self.topology = topo
        self.clamp_val = self.config.stability.curvature_clamp
        self.enable_trace_normalization = self.config.stability.enable_trace_normalization
        self.enable_trace_normalization = self.config.stability.enable_trace_normalization
        # Friction parameters are now handled by PhysicsEngine to avoid duplication

        # Feature dimension for gate input (Fourier for torus)
        gate_input_dim = dim * 2 if topo == TOPOLOGY_TORUS else dim

        # Low-rank basis parameters - initialized with small noise to break symmetry
        if num_heads > 1:
            self.U = nn.Parameter(torch.randn(num_heads, dim, rank) * 1e-4)
            self.W = nn.Parameter(torch.randn(num_heads, dim, rank) * 1e-4)
        else:
            self.U = nn.Parameter(torch.randn(dim, rank) * 1e-4)
            self.W = nn.Parameter(torch.randn(dim, rank) * 1e-4)

        # Friction gate
        friction_mode = getattr(self.config.stability, 'friction_mode', 'static')
        self.friction_gate = FrictionGate(dim, gate_input_dim, mode=friction_mode, num_heads=num_heads)

        # CONTRACT: LowRank ALWAYS returns (gamma_christoffel, mu_friction) separately.
        # The physics engine is the single authority on when/how friction is applied.
        # This prevents the P0.1 double-friction bug (geometry + engine both applying mu*v).
        self.return_friction_separately = True

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convert position to gate input features."""
        if self.topology == TOPOLOGY_TORUS:
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x

    def connection(self, v: torch.Tensor, w: torch.Tensor,
                   x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Bilinear Christoffel contraction: Γ(v,w)^k
        Γ^k_ij ≈ Σ_r W[r,k] * (U[i,r] * U[j,r])
        """
        if v.dim() == 3 and self.U.dim() == 3:
            v_r = torch.einsum('bhd, hdr -> bhr', v, self.U)
            w_r = torch.einsum('bhd, hdr -> bhr', w, self.U)
            vw_r = v_r * w_r
            gamma = torch.einsum('bhr, hdr -> bhd', vw_r, self.W)
        else:
            v_r = v @ self.U   # [..., rank]  (works for both 2D and 3D U)
            w_r = w @ self.U
            vw_r = v_r * w_r
            W_t = self.W.transpose(-1, -2) if self.W.dim() == 3 else self.W.t()
            gamma = vw_r @ W_t
        return torch.clamp(gamma, -self.clamp_val, self.clamp_val)

    def _normalize(self, gamma: torch.Tensor) -> torch.Tensor:
        """Symmetry-preserving trace normalization."""
        if gamma.dim() < 2:
            return gamma

        is_multi_head = (gamma.dim() == 3 and self.num_heads > 1)

        # Matrix case [..., D, D]
        if not is_multi_head and gamma.dim() >= 3 and gamma.shape[-1] == gamma.shape[-2]:
            gamma_sym = 0.5 * (gamma + gamma.transpose(-1, -2))
            if self.enable_trace_normalization:
                diag_mean = torch.diagonal(gamma_sym, dim1=-1, dim2=-2).mean(dim=-1, keepdim=True)
                correction = torch.diag_embed(diag_mean.expand(-1, self.dim))
                return gamma_sym - correction
            return gamma_sym

        # Vector case [..., D]
        if self.enable_trace_normalization:
            return gamma - gamma.mean(dim=-1, keepdim=True)
        return gamma

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        original_shape = v.shape
        # Handle multi-head [B, H, HD] → reshape to [B*H, HD] for matmul with U=[HD, rank]
        if v.dim() == 3 and self.U.dim() == 2:
            B, H, HD = v.shape
            v_flat = v.reshape(B * H, HD)   # [B*H, HD]
            x_flat = x.reshape(B * H, HD)
        else:
            v_flat = v
            x_flat = x
        B, H, HD = None, None, v_flat.shape[-1]
        R = self.rank
        
        # Check if we can take the fast CUDA path
        use_cuda_fused = (
            CUDA_AVAILABLE and 
            low_rank_christoffel_fwd is not None and 
            v_flat.is_cuda and 
            v_flat.dtype == torch.float32 and
            self.W.dim() == 3
        )

        if use_cuda_fused:
            # Reshape [B*H, HD] -> [B, H, HD] to match what kernel expects
            actual_B = original_shape[0] if v.dim() == 3 else 1
            actual_H = self.num_heads
            
            v_re = v_flat.view(actual_B, actual_H, HD)
            U_re = self.U.view(actual_H, HD, R)  # self.U is [H, D, R]
            W_re = self.W.view(actual_H, HD, R)
            
            gamma_re = LowRankChristoffelFunction.apply(
                v_re, U_re, W_re, self.clamp_val, self.enable_trace_normalization, False
            )
            gamma = gamma_re.view_as(v_flat)
        else:
            # Christoffel symbols via self-connection (Native Fallback)
            if v_flat.dim() == 3 and self.U.dim() == 3:
                v_r = torch.einsum('bhd, hdr -> bhr', v_flat, self.U)
                sq = v_r * v_r
                gamma = torch.einsum('bhr, hdr -> bhd', sq, self.W)
            else:
                v_r = v_flat @ self.U    # [..., rank]
                sq = v_r * v_r
                W_t = self.W.transpose(-1, -2) if self.W.dim() == 3 else self.W.t()
                gamma = sq @ W_t         # [..., HD]

        # Friction coefficient (position-dependent, gated)
        x_in = self._get_features(x_flat)
        mu = self.friction_gate(x_in, force=force)
        
        # Note: PhysicsEngine will add base friction and apply velocity scaling

        # Friction and normalizing only if not already done in CUDA
        if not use_cuda_fused:
            gamma = self._normalize(gamma)
            gamma = self.clamp_val * torch.tanh(gamma / self.clamp_val)

        # Restore original shape if we reshaped
        if B is not None:
            gamma = gamma.view(original_shape)
            mu = mu.view(original_shape)

        # CONTRACT: always return (gamma_pure, mu) so engine has single authority over friction
        return gamma, mu

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implicit Riemannian metric from the low-rank decomposition.
        
        The Christoffel parametrization Γ^k_ij ≈ Σ_r W_rk (U_ir U_jr) implies
        an underlying metric: g_ij ≈ Σ_r U_ir * U_jr = diag(U @ Uᵀ)
        
        Returns per-coordinate metric scale [..., D] or ones if shape unknown.
        T = (1/2) Σ_i g_ii v_i²  (Riemannian kinetic energy)
        """
        if self.U.dim() == 2:
            # Single head: U is [D, rank] → g_diag is [D]
            g_diag = (self.U ** 2).sum(dim=-1)  # [D]
            # Broadcast to x shape: handles [B, D], [B*H, D], any [..., D]
            return g_diag.expand_as(x)
        else:
            # Multi-head: U is [H, D, rank] → g_diag is [H, D]
            g_diag = (self.U ** 2).sum(dim=-1)  # [H, D]
            if x.dim() == 3 and x.shape[1] == self.num_heads:
                # [B, H, D]: structured multi-head
                return g_diag.unsqueeze(0).expand_as(x)
            else:
                # [B, H*D]: flat layout — expand g_diag [H,D] → [H*D], broadcast to [B, H*D]
                g_flat = g_diag.reshape(-1)  # [H*D]
                return g_flat.expand(x.shape[0], -1) if x.dim() == 2 else g_flat.expand_as(x)

    def dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.topology == TOPOLOGY_TORUS:
            diff = x1 - x2
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            return torch.norm(diff, dim=-1)
        return torch.norm(x1 - x2, dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.topology == TOPOLOGY_TORUS:
            return torch.atan2(torch.sin(x), torch.cos(x))
        return x


@register_geometry('low_rank_paper')
class PaperLowRankRiemannianGeometry(LowRankRiemannianGeometry):
    def __init__(self, dim: int, rank: int = 16, num_heads: int = 1,
                 config: Optional[PhysicsConfig] = None):
        super().__init__(dim, rank, num_heads=num_heads, config=config)
        self.return_friction_separately = True

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None,
                force: Optional[torch.Tensor] = None, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if v is None:
            return torch.zeros_like(x)

        original_shape = v.shape
        if v.dim() == 3 and self.U.dim() == 2:
            B, H, HD = v.shape
            v_flat = v.reshape(B * H, HD)
            x_flat = x.reshape(B * H, HD)
        else:
            v_flat = v
            x_flat = x
        B, H, HD = None, None, v_flat.shape[-1]
        R = self.rank

        use_cuda_fused = (
            CUDA_AVAILABLE and 
            low_rank_christoffel_fwd is not None and 
            v_flat.is_cuda and 
            v_flat.dtype == torch.float32 and
            self.W.dim() == 3
        )

        if use_cuda_fused:
            actual_B = original_shape[0] if v.dim() == 3 else 1
            actual_H = self.num_heads
            v_re = v_flat.view(actual_B, actual_H, HD)
            U_re = self.U.view(actual_H, HD, R)
            W_re = self.W.view(actual_H, HD, R)
            
            gamma_re = LowRankChristoffelFunction.apply(
                v_re, U_re, W_re, self.clamp_val, self.enable_trace_normalization, True
            )
            gamma = gamma_re.view_as(v_flat)
        else:
            if v_flat.dim() == 3 and self.U.dim() == 3:
                v_r = torch.einsum('bhd, hdr -> bhr', v_flat, self.U)
                denom = 1.0 + torch.norm(v_r, dim=-1, keepdim=True)
                phi = (v_r * v_r) / denom
                gamma = torch.einsum('bhr, hdr -> bhd', phi, self.W)
            else:
                v_r = v_flat @ self.U
                denom = 1.0 + torch.norm(v_r, dim=-1, keepdim=True)
                phi = (v_r * v_r) / denom
                W_t = self.W.transpose(-1, -2) if self.W.dim() == 3 else self.W.t()
                gamma = phi @ W_t

        x_in = self._get_features(x_flat)
        mu = self.friction_gate(x_in, force=force)

        if not use_cuda_fused:
            gamma = self._normalize(gamma)
            gamma = self.clamp_val * torch.tanh(gamma / self.clamp_val)

        if B is not None:
            gamma = gamma.view(original_shape)
            mu = mu.view(original_shape)

        # CONTRACT: Always return (gamma_pure, mu) for unified PhysicsEngine handling.
        return gamma, mu
