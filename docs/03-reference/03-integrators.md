# Integrator Reference

## Overview

Integrators are algorithms that numerically solve the system's differential equations. The choice of integrator affects accuracy, stability, and speed. Manifold implements multiple integrators for different use cases.

## Symplectic Integrators

Symplectic integrators preserve the Hamiltonian structure of the system, resulting in better energy preservation and more stable long-term trajectories.

### Leapfrog (Störmer–Verlet)

This is the main integrator and the most used. It is second order, which means the per-step error scales as O(h²). However, the energy error scales as O(h³), which makes it exceptional for Hamiltonian systems.

The algorithm has three phases per step:
1. Actualización parcial del momento (kick)
2. Actualización de la posición (drift)
3. Actualización parcial del momento (kick)

The Manifold implementation adds friction implicitly for better numerical stability:

v_new = (v + h * force) / (1 + h * mu + EPSILON)

This implicit form prevents the denominator from approaching zero when friction is significant.

Configurable parameters:
- DEFAULT_DT: timestep
- FRICTION_SCALE: friction coefficient
- LEAPFROG_SUBSTEPS: iterations per token step

### Yoshida

A fourth-order integrator that uses a sequence of 10 steps with optimized coefficients. The error scales as O(h⁵), significantly better than Leapfrog for the same timestep.

Better accuracy has a cost: more operations per step and higher memory usage to store intermediate states.

Useful when:
- High trajectory precision is required
- The timestep must be large
- Computational cost is not critical

Not recommended for production due to cost, but useful for benchmarking and validation.

### Forest-Ruth

Another fourth-order integrator, an alternative to Yoshida with a different structure. Less common in the literature but implemented for completeness.

The characteristics are similar to Yoshida: O(h⁵) error but more operations. The choice between Yoshida and Forest-Ruth is mostly preference.

### Omelyan

A second-order symplectic integrator optimized to minimize phase error. It combines Leapfrog simplicity with coefficients that reduce energy drift.

Useful when you need Leapfrog with better energy preservation, especially for long simulations.

## Runge-Kutta Integrators

Runge-Kutta integrators are generic methods for ODEs. They do not preserve symplectic structure but can be more accurate for non-Hamiltonian problems.

### Euler

The simplest integrator: y(t+h) = y(t) + h * f(y,t).

It does not preserve energy and diverges quickly. Included only for demonstration and testing, not for production use.

### Heun (Improved Euler)

Second-order method with predictor-corrector. More stable than Euler but less than Leapfrog for Hamiltonian systems.

Useful when friction dominates and symplectic structure is less important.

### Runge-Kutta 4 (RK4)

Standard fourth-order method. Accurate but expensive and does not preserve symplectic structure.

Useful for problems where accuracy is critical and the timestep is small.

### Dormand-Prince (RK45)

Adaptive integrator that adjusts the timestep based on estimated error. Useful for problems where the time scale varies significantly.

## Comparison Table

| Integrator | Order | Symplectic Preservation | Relative Cost | Recommended Use |
|------------|-------|-------------------------|----------------|-----------------|
| Leapfrog   | 2     | Yes                     | 1.0x           | Production      |
| Yoshida    | 4     | Yes                     | ~3.0x          | High precision  |
| Forest-Ruth| 4     | Yes                     | ~3.0x          | High precision  |
| Omelyan    | 2     | Yes                     | ~1.2x          | Better Leapfrog |
| Heun       | 2     | No                      | ~1.2x          | High friction   |
| RK4        | 4     | No                      | ~4.0x          | Benchmarking    |
| Dormand-Prince | 4+ | No                      | Variable       | Variable scales |

## Integrator Selection

For most tasks, Leapfrog is the right choice. It balances accuracy, stability, and efficiency.

Use Yoshida or Forest-Ruth when:
- The timestep must be large
- Trajectory precision is critical
- Computational cost is acceptable

Use Heun when:
- Friction dominates Hamiltonian dynamics
- Smooth transitions are needed

Avoid Euler in production. Its instability can cause divergence.

## Default Configuration

The system uses Leapfrog with the following default parameters:
- DEFAULT_DT = 0.05
- LEAPFROG_SUBSTEPS = 3
- FRICTION_SCALE = 0.02

This configuration was chosen to balance stability and exploration.

## Common Errors

If the loss diverges:
- Reduce DEFAULT_DT
- Increase FRICTION_SCALE
- Increase LEAPFROG_SUBSTEPS

If the loss converges too slowly:
- Increase DEFAULT_DT (up to 0.1)
- Reduce FRICTION_SCALE (down to 0.001)
- Reduce LEAPFROG_SUBSTEPS (to 1-2)

If you observe persistent oscillations:
- Friction may be too low
- Consider using Heun instead of Leapfrog
- Reduce IMPULSE_SCALE

---

**Manifold Labs (Joaquín Stürtz)**
