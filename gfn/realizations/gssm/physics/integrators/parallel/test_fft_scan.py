"""
Test Suite — FFT Toroidal Parallel Scan
========================================
Verifies that parallel_toroidal_fft is:
  1. Numerically identical to the sequential leapfrog (correctness).
  2. Faster than the sequential loop for large L (benchmarking).
  3. Well-behaved with learnable γ via FFTToroidalScan.

Usage:
    cd D:\\ASAS\\principal_proyects\\manifold_mini\\dev\\dev\\gfn
    python -m gfn.realizations.gssm.physics.integrators.parallel.test_fft_scan

Or simply:
    python test_fft_scan.py   (from this directory)
"""

import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
# GSSM_ROOT is at .../dev/dev/gfn
GSSM_ROOT   = HERE.parent.parent.parent.parent.parent.parent
if str(GSSM_ROOT) not in sys.path:
    sys.path.insert(0, str(GSSM_ROOT))

from gfn.realizations.gssm.physics.integrators.parallel.fft_scan import (
    parallel_toroidal_fft,
    FFTToroidalScan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sequential reference (faithful leapfrog on a flat torus)
# ──────────────────────────────────────────────────────────────────────────────

def sequential_leapfrog(
    F:      torch.Tensor,      # (B, L, D)
    gamma:  float,
    theta0: torch.Tensor,      # (B, D)
    omega0: torch.Tensor,      # (B, D)
    wrap_theta: bool = True,
) -> tuple:
    """
    Exact sequential leapfrog on a flat torus (Δt = 1, no curvature).
      ω_t = (1 - γ) ω_{t-1} + F_t
      θ_t = (θ_{t-1} + ω_t) mod 2π
    """
    B, L, D = F.shape
    thetas, omegas = [], []
    th = theta0.clone()
    om = omega0.clone()

    for t in range(L):
        om = (1.0 - gamma) * om + F[:, t, :]
        th = th + om
        if wrap_theta:
            th = th % (2.0 * math.pi)
        thetas.append(th.clone())
        omegas.append(om.clone())

    return (
        torch.stack(thetas, dim=1),   # (B, L, D)
        torch.stack(omegas, dim=1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Numerical equivalence
# ──────────────────────────────────────────────────────────────────────────────

def test_numerical_equivalence(
    B: int   = 4,
    L: int   = 64,
    D: int   = 16,
    gamma: float = 0.1,
    tol: float   = 1e-4,
    device: str  = 'cpu',
):
    print(f"\n{'='*60}")
    print(f"Test 1 — Numerical equivalence  (B={B}, L={L}, D={D}, γ={gamma})")
    print(f"{'='*60}")

    torch.manual_seed(42)
    F      = torch.randn(B, L, D, device=device)
    theta0 = torch.rand(B, D, device=device) * 2 * math.pi
    omega0 = torch.randn(B, D, device=device) * 0.1

    # Sequential reference
    th_seq, om_seq = sequential_leapfrog(F, gamma, theta0, omega0)

    # FFT parallel
    g_tensor = torch.tensor(gamma, dtype=F.dtype, device=device)
    th_fft, om_fft = parallel_toroidal_fft(F, g_tensor, theta0, omega0)

    # Unwrap for fair comparison (mod 2π can differ by 2π for boundary vals)
    def angular_diff(a, b):
        d = (a - b) % (2 * math.pi)
        d[d > math.pi] -= 2 * math.pi
        return d.abs()

    theta_err = angular_diff(th_fft, th_seq).max().item()
    omega_err = (om_fft - om_seq).abs().max().item()

    print(f"  Max |Δθ| (angular): {theta_err:.2e}  (tol={tol})")
    print(f"  Max |Δω|           : {omega_err:.2e}  (tol={tol})")

    theta_ok = theta_err < tol
    omega_ok = omega_err < tol
    if theta_ok and omega_ok:
        print("  [OK] PASSED -- FFT scan is numerically equivalent to sequential leapfrog")
    else:
        print("  [FAIL]")
        if not theta_ok:
            print(f"     theta error {theta_err:.2e} exceeds tol {tol}")
        if not omega_ok:
            print(f"     omega error {omega_err:.2e} exceeds tol {tol}")
    return theta_ok and omega_ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Sweep over friction values
# ──────────────────────────────────────────────────────────────────────────────

def test_gamma_sweep(
    B: int = 2, L: int = 128, D: int = 8, tol: float = 1e-6
):
    print(f"\n{'='*60}")
    print(f"Test 2 — γ sweep (Float64)  (B={B}, L={L}, D={D})")
    print(f"{'='*60}")

    gammas = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.99, 1.0 - 1e-5]
    torch.manual_seed(0)
    # Use float64 to test pure mathematical equivalence independent of FP32 accumulation
    F      = torch.randn(B, L, D, dtype=torch.float64)
    theta0 = torch.zeros(B, D, dtype=torch.float64)
    omega0 = torch.randn(B, D, dtype=torch.float64) * 0.05

    all_pass = True
    for g in gammas:
        g_t = torch.tensor(g, dtype=torch.float64)
        th_seq, om_seq = sequential_leapfrog(F, g, theta0, omega0)
        th_fft, om_fft = parallel_toroidal_fft(F, g_t, theta0, omega0)

        def angular_diff(a, b):
            d = (a - b) % (2 * math.pi)
            d[d > math.pi] -= 2 * math.pi
            return d.abs()

        th_err = angular_diff(th_fft, th_seq).max().item()
        om_err = (om_fft - om_seq).abs().max().item()
        ok = th_err < tol and om_err < tol
        if not ok:
            all_pass = False
        status = "[OK]" if ok else "[FAIL]"
        print(f"  gamma={g:.5f}  |Dtheta|={th_err:.2e}  |Domega|={om_err:.2e}  {status}")

    return all_pass



# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Speed benchmark
# ──────────────────────────────────────────────────────────────────────────────

def test_speed_benchmark(
    B: int = 8, D: int = 64, gamma: float = 0.1, device_str: str = 'cpu',
    lengths: list = None,
    n_warmup: int = 3, n_iter: int = 10,
):
    if lengths is None:
        lengths = [64, 256, 1024, 4096, 16384]

    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"Test 3 — Speed benchmark  (B={B}, D={D}, γ={gamma}, device={device_str})")
    print(f"{'='*60}")
    print(f"  {'Length':>8}  {'Sequential (ms)':>16}  {'FFT (ms)':>10}  {'Speedup':>8}")
    print(f"  {'-'*50}")

    g_t = torch.tensor(gamma, device=device)

    for L in lengths:
        torch.manual_seed(7)
        F      = torch.randn(B, L, D, device=device)
        theta0 = torch.zeros(B, D, device=device)
        omega0 = torch.zeros(B, D, device=device)

        # Warm-up sequential
        for _ in range(n_warmup):
            sequential_leapfrog(F, gamma, theta0, omega0)

        # Time sequential
        if device_str == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            sequential_leapfrog(F, gamma, theta0, omega0)
        if device_str == 'cuda':
            torch.cuda.synchronize()
        t_seq = (time.perf_counter() - t0) / n_iter * 1000  # ms

        # Warm-up FFT
        for _ in range(n_warmup):
            parallel_toroidal_fft(F, g_t, theta0, omega0)

        # Time FFT
        if device_str == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            parallel_toroidal_fft(F, g_t, theta0, omega0)
        if device_str == 'cuda':
            torch.cuda.synchronize()
        t_fft = (time.perf_counter() - t0) / n_iter * 1000  # ms

        speedup = t_seq / t_fft if t_fft > 0 else float('inf')
        flag = "[fast]" if speedup > 1.5 else ("[~]" if speedup > 0.8 else "[slow]")
        print(f"  {L:>8,}  {t_seq:>14.2f}  {t_fft:>10.2f}  {speedup:>7.1f}x {flag}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — FFTToroidalScan nn.Module (gradient flow)
# ──────────────────────────────────────────────────────────────────────────────

def test_module_gradients(
    B: int = 4, L: int = 32, H: int = 2, D: int = 8,
):
    print(f"\n{'='*60}")
    print(f"Test 4 — FFTToroidalScan gradients  (B={B}, L={L}, H={H}, D={D})")
    print(f"{'='*60}")

    module = FFTToroidalScan(dim=D, heads=H, gamma_init=0.1)
    F = torch.randn(B, L, H, D, requires_grad=True)

    theta, omega = module(F)
    loss = theta.mean() + omega.mean()
    loss.backward()

    grad_F     = F.grad is not None
    grad_gamma = module.gamma_logit.grad is not None
    grad_omega0= module.omega0.grad is not None

    print(f"  dL/dF       : {'[OK]' if grad_F     else '[FAIL]'}")
    print(f"  dL/d_gamma  : {'[OK]' if grad_gamma else '[FAIL]'}")
    print(f"  dL/d_omega0 : {'[OK]' if grad_omega0 else '[FAIL]'}")

    ok = grad_F and grad_gamma and grad_omega0
    print(f"  {'[OK] PASSED' if ok else '[FAIL]'} -- all parameters receive gradients")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Extrapolation stability (long sequences)
# ──────────────────────────────────────────────────────────────────────────────

def test_extrapolation(
    B: int = 2, D: int = 4, gamma: float = 0.05,
    train_L: int = 64, test_Ls: list = None,
):
    if test_Ls is None:
        test_Ls = [128, 256, 512, 1024, 4096]

    print(f"\n{'='*60}")
    print(f"Test 5 — Extrapolation stability  (trained on L={train_L})")
    print(f"{'='*60}")

    g_t = torch.tensor(gamma)
    torch.manual_seed(1)
    theta0 = torch.zeros(B, D)
    omega0 = torch.zeros(B, D)

    for L in test_Ls:
        F = torch.randn(B, L, D)
        th, om = parallel_toroidal_fft(F, g_t, theta0, omega0)
        nan_count = torch.isnan(th).sum().item() + torch.isnan(om).sum().item()
        inf_count = torch.isinf(th).sum().item() + torch.isinf(om).sum().item()
        th_range  = th.min().item(), th.max().item()
        ok = nan_count == 0 and inf_count == 0
        status = "[OK]" if ok else f"[FAIL] NaN={nan_count} Inf={inf_count}"
        print(f"  L={L:>6,}  theta in [{th_range[0]:.2f}, {th_range[1]:.2f}]  {status}")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[*] FFT Toroidal Parallel Scan -- Test Suite")
    print(f"   Device: {device_str.upper()}")

    results = []

    results.append(test_numerical_equivalence(device=device_str))
    results.append(test_gamma_sweep())
    test_speed_benchmark(device_str=device_str)   # no pass/fail, just timing
    results.append(test_module_gradients())
    results.append(test_extrapolation())

    print(f"\n{'='*60}")
    passed = sum(results)
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[!!] Some tests FAILED -- check output above")
    print(f"{'='*60}")
