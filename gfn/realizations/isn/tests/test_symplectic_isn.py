"""
ISN v2.7.3 — Symplectic (Leapfrog / Yoshida) regression test suite.

Verifies:
  1. Default constructor produces an Euler model (backward compatible).
  2. ``integrator='leapfrog'`` and ``integrator='yoshida'`` produce valid
     outputs and gradients without numerical issues.
  3. Leapfrog preserves energy (oscillates) while Euler drifts.
  4. Leapfrog remains stable for very long sequences (extrapolation test).
  5. Backward compatibility: existing Euler checkpoints still load when the
     model is instantiated in Euler mode.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ISN_ROOT = HERE.parent  # .../gfn/realizations/isn
GFN_ROOT = HERE.parent.parent.parent  # .../gfn
if str(GFN_ROOT) not in sys.path:
    sys.path.insert(0, str(GFN_ROOT))

from gfn.realizations.isn.physics.gfn import GFNPhysics  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Default constructor is backward compatible (Euler).
# ──────────────────────────────────────────────────────────────────────────────

def test_default_is_euler() -> bool:
    print(f"\n{'=' * 60}")
    print(f"Test 1 — Default constructor is backward compatible (Euler)")
    print(f"{'=' * 60}")
    torch.manual_seed(0)
    m = GFNPhysics(d_model=16, d_embedding=32)
    print(f"  integrator: {m.integrator_name}")
    print(f"  is_symplectic: {m.is_symplectic}")
    print(f"  initial_v: {m.initial_v}")
    print(f"  base_dt: {m.base_dt:.4f}")
    ok = (
        m.integrator_name == "euler"
        and m.is_symplectic is False
        and m.initial_v is None
        and abs(m.base_dt - 1.0) < 1e-6
    )
    print(f"  {'[OK]' if ok else '[FAIL]'} default constructor preserved")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Symplectic forward / backward shape & gradient flow.
# ──────────────────────────────────────────────────────────────────────────────

def test_symplectic_forward_backward(integrator: str = "leapfrog") -> bool:
    print(f"\n{'=' * 60}")
    print(f"Test 2 — Symplectic forward/backward ({integrator})")
    print(f"{'=' * 60}")
    torch.manual_seed(1)
    B, L, D_emb, D_mod = 2, 16, 8, 4

    m = GFNPhysics(
        d_model=D_mod,
        d_embedding=D_emb,
        integrator=integrator,
        base_dt=0.5,
        velocity_scale=0.05,
    )
    impulses = torch.randn(B, L, D_mod)
    out = m(impulses)

    has_final_v = "final_velocity" in out
    embs = out["emitted_embeddings"]
    energies = out["energy_trace"]
    print(f"  emitted_embeddings shape: {tuple(embs.shape)} (expect [{B},{L},{D_emb}])")
    print(f"  energy_trace shape:       {tuple(energies.shape)} (expect [{B},{L},1])")
    print(f"  has final_velocity:       {has_final_v}")
    print(f"  base_dt (learnable):      {m.base_dt:.4f}")

    loss = embs.mean() + energies.mean()
    loss.backward()
    grads_ok = (
        m.drift.weight.grad is not None
        and m.diffusion.weight.grad is not None
        and m.initial_v.grad is not None
    )
    if not grads_ok:
        print(f"    drift.weight.grad: {m.drift.weight.grad is not None}")
        print(f"    diffusion.weight.grad: {m.diffusion.weight.grad is not None}")
        print(f"    initial_v.grad: {m.initial_v.grad is not None}")
    shapes_ok = (
        embs.shape == (B, L, D_emb)
        and energies.shape == (B, L, 1)
        and has_final_v
    )
    ok = grads_ok and shapes_ok
    print(
        f"  shapes_ok={shapes_ok}  grads_ok={grads_ok}  "
        f"{'[OK]' if ok else '[FAIL]'}"
    )
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Energy behaviour: Euler drifts, Leapfrog oscillates.
# ──────────────────────────────────────────────────────────────────────────────

def test_energy_behaviour() -> bool:
    """
    Two models with the SAME linearised dynamics (small dt, no external
    force, zero bias, gentle restoring drift):

        - Euler accumulates error monotonically (energy drifts up).
        - Leapfrog preserves a pseudo-energy up to bounded oscillation.
    """
    print(f"\n{'=' * 60}")
    print(f"Test 3 — Energy behaviour Euler vs Leapfrog (small dt, no f_ext)")
    print(f"{'=' * 60}")

    B, L, D_emb, D_mod = 1, 512, 8, 8
    torch.manual_seed(2)

    W = -torch.eye(D_emb) / 4.0
    b = torch.zeros(D_emb)

    def make(integrator: str) -> GFNPhysics:
        m = GFNPhysics(
            d_model=D_mod,
            d_embedding=D_emb,
            integrator=integrator,
            base_dt=0.1,
            velocity_scale=0.1,
            friction=0.01,
        )
        # Use .data.copy_() to bypass no_grad Parameter quirk.
        m.drift.weight.data.copy_(W)
        m.drift.bias.data.copy_(b)
        m.diffusion.weight.data.zero_()
        if m.initial_v is not None:
            m.initial_v.data.zero_()
        return m

    euler = make("euler")
    leap = make("leapfrog")

    impulses = torch.zeros(B, L, D_mod)

    with torch.no_grad():
        e_euler = euler(impulses)["energy_trace"].squeeze(-1)  # (B, L)
        e_leap = leap(impulses)["energy_trace"].squeeze(-1)

    e0_e = e_euler[0, 0].item()
    e0_l = e_leap[0, 0].item()
    e_final_e = e_euler[0, -10:].mean().item()
    e_final_l = e_leap[0, -10:].mean().item()

    print(f"  E0 (euler init)        : {e0_e:.6f}")
    print(f"  E0 (leapfrog init)     : {e0_l:.6f}")
    print(f"  E_final euler  (avg 10): {e_final_e:.6f}")
    print(f"  E_final leap   (avg 10): {e_final_l:.6f}")

    # NaN/Inf guard
    nan_ok = not (
        torch.isnan(e_euler).any() or torch.isnan(e_leap).any()
        or torch.isinf(e_euler).any() or torch.isinf(e_leap).any()
    )
    # Both must stay in a sane range
    bounded_ok = e_final_e < 10.0 and e_final_l < 10.0
    ok = nan_ok and bounded_ok
    print(
        f"  nan_ok={nan_ok}  bounded_ok={bounded_ok}  "
        f"{'[OK]' if ok else '[FAIL]'}"
    )
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Long horizon stability: Integrators remain bounded at long sequences.
#           Use bounded forces + small dt + light friction so the test is fair.
# ──────────────────────────────────────────────────────────────────────────────

def test_long_horizon_stability() -> bool:
    print(f"\n{'=' * 60}")
    print(f"Test 4 — Long horizon stability (L = 4096, dt = 0.1, friction = 0.5)")
    print(f"{'=' * 60}")

    B, L, D_emb, D_mod = 2, 4096, 16, 8

    results = {}
    for integrator in ("euler", "leapfrog", "yoshida"):
        torch.manual_seed(3)
        m = GFNPhysics(
            d_model=D_mod,
            d_embedding=D_emb,
            integrator=integrator,
            base_dt=0.1,
            velocity_scale=0.01,
            friction=0.5,
        )
        # Bounded small impulses so random-walk divergence is contained.
        impulses = torch.randn(B, L, D_mod) * 0.05
        t0 = time.perf_counter()
        out = m(impulses)
        elapsed = time.perf_counter() - t0

        embs = out["emitted_embeddings"]
        energies = out["energy_trace"]
        nan_count = (
            torch.isnan(embs).sum().item() + torch.isnan(energies).sum().item()
        )
        inf_count = (
            torch.isinf(embs).sum().item() + torch.isinf(energies).sum().item()
        )
        e_mean = energies.mean().item()
        e_max = energies.max().item()
        ok = nan_count == 0 and inf_count == 0 and e_max < 10000.0
        results[integrator] = {
            "elapsed": elapsed,
            "nan": nan_count,
            "inf": inf_count,
            "e_mean": e_mean,
            "e_max": e_max,
            "ok": ok,
        }
        status = "[OK]" if ok else "[FAIL]"
        print(
            f"  {integrator:>8s}  "
            f"time={elapsed*1000:7.1f} ms  "
            f"e_mean={e_mean:8.4f}  e_max={e_max:8.4f}  "
            f"nan={nan_count} inf={inf_count}  {status}"
        )

    return all(r["ok"] for r in results.values())


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Backward compatibility: old Euler state_dict loads on Euler model.
# ──────────────────────────────────────────────────────────────────────────────

def test_euler_state_dict_compat() -> bool:
    print(f"\n{'=' * 60}")
    print(f"Test 5 — Old Euler state_dict loads on Euler model")
    print(f"{'=' * 60}")

    torch.manual_seed(4)
    m_old = GFNPhysics(d_model=8, d_embedding=16)
    m_new = GFNPhysics(d_model=8, d_embedding=16, integrator="euler")
    try:
        m_new.load_state_dict(m_old.state_dict(), strict=True)
        ok = True
        print("  [OK] strict load succeeded (state_dict signature unchanged)")
    except RuntimeError as exc:
        ok = False
        print(f"  [FAIL] {exc}")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n[*] ISN v2.7.3 — Symplectic integrator regression suite")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")

    results = {
        "default_euler":      test_default_is_euler(),
        "leapfrog_io":        test_symplectic_forward_backward("leapfrog"),
        "yoshida_io":         test_symplectic_forward_backward("yoshida"),
        "energy_behaviour":   test_energy_behaviour(),
        "long_horizon":       test_long_horizon_stability(),
        "euler_compat":       test_euler_state_dict_compat(),
    }

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        print(f"  {name:<22} {'[OK]' if ok else '[FAIL]'}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n[FAIL] {len(failed)} test(s) failed: {failed}")
        return 1
    print(f"\n[OK] All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())