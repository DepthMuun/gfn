# GSSM Integrators

This directory contains detailed mathematical explanations of all integrators available in GSSM.

---

## Available Integrators

### Symplectic (Energy-Preserving)

| File | Name | Order | Force Evals/Step | Best For |
|------|------|-------|------------------|----------|
| `leapfrog.md` | Leapfrog / Störmer-Verlet | 2nd | 2 | **Default - Training** |
| `verlet.md` | Velocity Verlet | 2nd | 2 | Position-velocity sync |
| `yoshida.md` | Yoshida | 4th | 3 | Long simulations |
| `forest_ruth.md` | Forest-Ruth | 4th | 3 | Alternative 4th order |
| `omelyan.md` | Omelyan | 2nd | 6 | Optimized accuracy |

### Runge-Kutta (Accuracy-Focused)

| File | Name | Order | Force Evals/Step | Best For |
|------|------|-------|------------------|----------|
| `rk4.md` | Runge-Kutta 4 | 4th | 4 | Short trajectories |
| `heun.md` | Heun / Improved Euler | 2nd | 2 | General ODEs |

---

## Quick Selection Guide

**For Training:**
- Use **Leapfrog** (default) - most stable

**For Long Sequences:**
- Use **Yoshida** or **Forest-Ruth** (4th order symplectic)

**For Validation:**
- Use **RK4** (high accuracy)

**For Quick Tests:**
- Use **Heun** (simple)

---

## What is a Symplectic Integrator?

A symplectic integrator preserves the symplectic 2-form in phase space:

$$\omega = dp \wedge dq$$

This means:
- Phase space volume is conserved
- Energy oscillates but doesn't drift
- Good for long-term Hamiltonian dynamics

Non-symplectic methods (like RK4) have systematic energy drift over time.

---

## Comparison Summary

| Method | Order | Symplectic | Cost | Accuracy | Energy |
|--------|-------|------------|------|----------|--------|
| Leapfrog | 2 | ✅ | 1× | Good | Conserved |
| Yoshida | 4 | ✅ | 3× | High | Conserved |
| RK4 | 4 | ❌ | 4× | Very High | Drifts |
| Heun | 2 | ❌ | 1× | Good | Drifts |

---

## Reading Order

1. Start with **Leapfrog** (most important, default)
2. Then **Yoshida** (for 4th order)
3. Then **RK4** (non-symplectic alternative)
4. Others as needed

---

*Last Updated: 2026-04-02*
