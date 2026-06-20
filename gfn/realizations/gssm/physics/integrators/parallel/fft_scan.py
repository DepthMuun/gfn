"""
FFT Toroidal Parallel Scan — GFN / GSSM
========================================
Implements the closed-form parallel solution for the toroidal recurrence
derived in paral.md.

The sequential recurrence (Leapfrog on a FLAT torus, Δt=1):
    ω_t = (1 - γ) ω_{t-1} + F_t
    θ_t = (θ_{t-1} + ω_t) mod 2π

State vector  S_t = [θ_t, ω_t]^T satisfies:
    S_t = A · S_{t-1} + B · F_t   (mod M)

where  A = [[1, 1-γ], [0, 1-γ]],   B = [[1], [1]],   M = [2π, ∞]

Closed-form matrix power (flat torus → zero curvature → constant coefficients):
    A^k = [[1,  α(k)], [0, (1-γ)^k]]
    α(k) = ((1-γ) - (1-γ)^{k+1}) / γ    (sum of geometric series)

Full parallel solution for all t in [1, L]:
    S_t = A^t S_0 + Σ_{i=1}^{t} A^{t-i} B F_i   (mod M)

The convolution sum Σ_{i=1}^{t} A^{t-i} B F_i is a causal 1-D linear
convolution with the kernel K_k = A^k B, parallelised via FFT.

Complexity: O(L log L) with a single rfft/irfft pair.
Memory:     O(L · D)  — no per-step hidden state storage.

This replaces the O(L) sequential loop in ManifoldLayer / ToroidalFlowChannel.

Author: generated for MANIFOLD/DTT research
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Low-level kernel
# ──────────────────────────────────────────────────────────────────────────────

def _build_kernel(gamma: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the two causal decay kernels for lengths k = 0 … L-1.

    For a given input force F_i at time i, it affects:
    - omega_t for t >= i via (1-gamma)^(t-i) * F_i
    - theta_t for t >= i via sum_{j=i}^t (1-gamma)^(j-i) * F_i
      which is sum_{k=0}^{t-i} (1-gamma)^k * F_i.

    Let k = t - i.
    K_omega_k = (1-gamma)^k
    K_theta_k = sum_{j=0}^k (1-gamma)^j = (1 - (1-gamma)^(k+1)) / gamma
    """
    device = gamma.device
    dtype  = gamma.dtype
    k      = torch.arange(L, device=device, dtype=dtype)            # (L,)

    decay  = (1.0 - gamma) ** k                                     # (1-γ)^k

    eps    = 1e-7
    safe_g = gamma.clamp(min=eps)
    # K_theta_k = (1 - (1-gamma)^(k+1)) / gamma
    alpha  = (1.0 - decay * (1.0 - gamma)) / safe_g

    # For gamma ≈ 0: alpha_k = k + 1
    # We can handle this smoothly:
    is_zero = (gamma < eps)
    if is_zero.any():
        alpha_zero = k + 1.0
        alpha = torch.where(is_zero, alpha_zero, alpha)

    return decay, alpha                                             # each: (L,)



def parallel_toroidal_fft(
    F:     torch.Tensor,          # (B, L, D) or (B, L, H, D) — external forces
    gamma: torch.Tensor,          # scalar or (D,) or (H, D)  — friction per dim
    theta0: Optional[torch.Tensor] = None,  # (B, D) or (B, H, D) — initial position
    omega0: Optional[torch.Tensor] = None,  # (B, D) or (B, H, D) — initial velocity
    wrap_theta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ALL positions θ_t and velocities ω_t for t = 1 … L in O(L log L).

    The computation is exact (to floating-point precision) and numerically
    equivalent to the sequential leapfrog on a flat torus.

    Args:
        F        : Force sequence. Shape (B, L, D) or (B, L, H, D).
        gamma    : Friction coefficient(s). Must broadcast against F's last dim(s).
        theta0   : Initial positions. Defaults to zeros.
        omega0   : Initial velocities. Defaults to zeros.
        wrap_theta: If True, wraps θ ∈ [0, 2π). Set False for Euclidean coords.

    Returns:
        theta : same shape as F — positions at each step.
        omega : same shape as F — velocities at each step.
    """
    shape_in = F.shape
    B, L     = shape_in[0], shape_in[1]
    D_rest   = shape_in[2:]                   # () or (H,) or (H, D)

    # ── Flatten trailing dims to a single D for the 1-D convolution ──────────
    F_flat  = F.reshape(B, L, -1)            # (B, L, D_total)
    D_total = F_flat.shape[-1]

    # ── Expand gamma to (D_total,) ────────────────────────────────────────────
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma, dtype=F.dtype, device=F.device)
    gamma = gamma.to(dtype=F.dtype, device=F.device).reshape(-1)
    if gamma.numel() == 1:
        gamma = gamma.expand(D_total)
    assert gamma.numel() == D_total, \
        f"gamma numel {gamma.numel()} must match D_total={D_total}"

    # ── Initial conditions ────────────────────────────────────────────────────
    if theta0 is None:
        theta0_flat = F_flat.new_zeros(B, D_total)
    else:
        theta0_flat = theta0.reshape(B, D_total).to(dtype=F.dtype)

    if omega0 is None:
        omega0_flat = F_flat.new_zeros(B, D_total)
    else:
        omega0_flat = omega0.reshape(B, D_total).to(dtype=F.dtype)

    # ── Build causal kernels  (one per dimension d) ───────────────────────────
    # K_omega[k, d] = (1-γ_d)^k,   K_theta[k, d] = α_d(k)
    # shape: (L, D_total)
    K_omega_list, K_theta_list = [], []
    for d in range(D_total):
        kw, kt = _build_kernel(gamma[d], L)
        K_omega_list.append(kw)
        K_theta_list.append(kt)
    K_omega = torch.stack(K_omega_list, dim=1)   # (L, D_total)
    K_theta = torch.stack(K_theta_list, dim=1)   # (L, D_total)

    # ── Causal convolution via FFT (Overlap-Add) ──────────────────────────────
    # We need conv_len ≥ 2L-1 to avoid circular aliasing in a causal conv.
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len <<= 1

    # F_flat: (B, L, D) → pad → (B, fft_len, D)
    F_pad = torch.zeros(B, fft_len, D_total, device=F.device, dtype=F.dtype)
    F_pad[:, :L, :] = F_flat

    # K: (L, D) → pad → (fft_len, D)
    Kw_pad = torch.zeros(fft_len, D_total, device=F.device, dtype=F.dtype)
    Kw_pad[:L, :] = K_omega
    Kt_pad = torch.zeros(fft_len, D_total, device=F.device, dtype=F.dtype)
    Kt_pad[:L, :] = K_theta

    # FFT along time axis (dim=1)
    F_fft  = torch.fft.rfft(F_pad,  n=fft_len, dim=1)   # (B, fft_len//2+1, D)
    Kw_fft = torch.fft.rfft(Kw_pad, n=fft_len, dim=0)   # (fft_len//2+1, D)
    Kt_fft = torch.fft.rfft(Kt_pad, n=fft_len, dim=0)   # (fft_len//2+1, D)

    # Multiply in frequency domain (broadcast over B)
    omega_fft = F_fft * Kw_fft.unsqueeze(0)              # (B, fft_len//2+1, D)
    theta_fft = F_fft * Kt_fft.unsqueeze(0)              # (B, fft_len//2+1, D)

    # Inverse FFT → causal convolution result
    omega_conv = torch.fft.irfft(omega_fft, n=fft_len, dim=1)[:, :L, :]   # (B, L, D)
    theta_conv = torch.fft.irfft(theta_fft, n=fft_len, dim=1)[:, :L, :]   # (B, L, D)

    # ── Add homogeneous solution (initial conditions) ─────────────────────────
    # ω_t^{hom} = (1-γ)^t · ω_0
    # θ_t^{hom} = θ_0 + α(t) · ω_0
    t_idx   = torch.arange(1, L + 1, device=F.device, dtype=F.dtype)       # (L,)

    # (L, D_total): decay^t per dim
    hom_decay  = (1.0 - gamma.unsqueeze(0)) ** t_idx.unsqueeze(1)           # (L, D)

    # α(t) · ω_0: same formula as kernel build but for indices 1..L
    safe_g    = gamma.clamp(min=1e-7)
    hom_alpha = (1.0 - hom_decay) * (1.0 - gamma.unsqueeze(0)) / safe_g    # (L, D)

    # Handle gamma ≈ 0: hom_alpha = t
    is_zero = (gamma.unsqueeze(0) < 1e-7).expand(L, -1)
    if is_zero.any():
        alpha_zero = t_idx.unsqueeze(1).expand(-1, D_total)
        hom_alpha = torch.where(is_zero, alpha_zero, hom_alpha)


    omega_hom = hom_decay.unsqueeze(0)  * omega0_flat.unsqueeze(1)          # (B, L, D)
    theta_hom = (
        theta0_flat.unsqueeze(1)                                            # (B, 1, D) → (B, L, D)
        + hom_alpha.unsqueeze(0) * omega0_flat.unsqueeze(1)
    )

    omega_out = omega_conv + omega_hom
    theta_out = theta_conv + theta_hom

    if wrap_theta:
        theta_out = theta_out % (2.0 * math.pi)

    # ── Restore original shape ────────────────────────────────────────────────
    theta_out = theta_out.reshape(shape_in)
    omega_out = omega_out.reshape(shape_in)

    return theta_out, omega_out


# ──────────────────────────────────────────────────────────────────────────────
# nn.Module wrapper — drop-in replacement for sequential ToroidalFlowChannel
# ──────────────────────────────────────────────────────────────────────────────

class FFTToroidalScan(nn.Module):
    """
    Parallel Toroidal Flow Channel using FFT Causal Convolution.

    Drop-in replacement for a sequential leapfrog loop on a flat torus.

    Key properties:
    - O(L log L) time, O(L·D) memory — no sequential dependency.
    - Trainable per-dimension friction γ_d ∈ (0, 1) and initial velocity ω_0.
    - Supports arbitrary batch/head dimensions.

    Args:
        dim      : feature dimension D.
        heads    : number of independent tori (heads). Default 1.
        gamma_init: initial friction value. Default 0.1.
        learn_gamma: if True, γ is a learnable parameter. Default True.
        learn_omega0: if True, per-dim initial velocity is learnable. Default True.
        wrap_theta: wrap θ mod 2π. Default True.
    """

    def __init__(
        self,
        dim:          int,
        heads:        int   = 1,
        gamma_init:   float = 0.1,
        learn_gamma:  bool  = True,
        learn_omega0: bool  = True,
        wrap_theta:   bool  = True,
    ):
        super().__init__()
        self.dim        = dim
        self.heads      = heads
        self.wrap_theta = wrap_theta

        # γ parameterized in logit space to stay in (0, 1)
        gamma_logit_init = math.log(gamma_init / (1.0 - gamma_init))
        if learn_gamma:
            self.gamma_logit = nn.Parameter(
                torch.full((heads, dim), gamma_logit_init)
            )
        else:
            self.register_buffer(
                'gamma_logit',
                torch.full((heads, dim), gamma_logit_init)
            )

        if learn_omega0:
            self.omega0 = nn.Parameter(torch.zeros(heads, dim))
        else:
            self.register_buffer('omega0', torch.zeros(heads, dim))

    @property
    def gamma(self) -> torch.Tensor:
        """Friction coefficients in (0, 1), shape (H, D)."""
        return torch.sigmoid(self.gamma_logit)

    def forward(
        self,
        F:      torch.Tensor,                       # (B, L, H, D) forces
        theta0: Optional[torch.Tensor] = None,      # (B, H, D) initial pos
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F      : force sequence (B, L, H, D).
            theta0 : optional initial position (B, H, D). Defaults to zeros.

        Returns:
            theta  : positions (B, L, H, D) on the torus.
            omega  : velocities (B, L, H, D).
        """
        B, L, H, D = F.shape
        assert H == self.heads and D == self.dim, \
            f"Expected (B,L,{self.heads},{self.dim}), got {F.shape}"

        # Expand ω_0 to batch: (H, D) → (B, H, D)
        omega0_b = self.omega0.unsqueeze(0).expand(B, -1, -1)   # (B, H, D)

        # γ: (H, D) — constant across batch and time (flat torus)
        gamma_hd = self.gamma                                    # (H, D)

        # We process each head independently but batch them by reshaping:
        # (B, L, H, D) → (B*H, L, D) so the FFT runs over all heads at once.
        F_bh   = F.permute(0, 2, 1, 3).reshape(B * H, L, D)   # (B*H, L, D)
        o0_bh  = omega0_b.reshape(B * H, D)                    # (B*H, D)
        t0_bh  = (theta0.reshape(B * H, D)
                  if theta0 is not None
                  else None)

        # gamma broadcast: (H, D) → (B*H, D) by repeating per batch
        gamma_bh = gamma_hd.unsqueeze(0).expand(B, -1, -1).reshape(B * H, D)



        # ── Vectorised batch-aware FFT scan ───────────────────────────────────
        # Rewrite: call the scan once per unique gamma (H unique gammas).
        theta_list, omega_list = [], []
        for h in range(H):
            F_h    = F[:, :, h, :]                                  # (B, L, D)
            o0_h   = omega0_b[:, h, :]                              # (B, D)
            t0_h   = (theta0[:, h, :] if theta0 is not None
                      else None)
            g_h    = gamma_hd[h]                                    # (D,)

            th_h, om_h = parallel_toroidal_fft(
                F      = F_h,
                gamma  = g_h,
                theta0 = t0_h,
                omega0 = o0_h,
                wrap_theta = self.wrap_theta,
            )
            theta_list.append(th_h)                                 # (B, L, D)
            omega_list.append(om_h)

        theta = torch.stack(theta_list, dim=2)                      # (B, L, H, D)
        omega = torch.stack(omega_list, dim=2)                      # (B, L, H, D)

        return theta, omega
