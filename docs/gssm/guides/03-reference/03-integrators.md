# Integrator Reference

## Overview

Integrators are algorithms that numerically solve the system's differential equations. The choice of integrator affects accuracy, stability, and speed. Manifold implements multiple integrators for different use cases, ranging from simple first-order methods to sophisticated fourth-order symplectic schemes.

## Symplectic Integrators

Symplectic integrators preserve the Hamiltonian structure of the system, resulting in better energy preservation and more stable long-term trajectories. These integrators are particularly well-suited for Hamiltonian systems where energy conservation is important.

### Leapfrog (Störmer-Verlet)

This is the main integrator and the most widely used. It is second-order accurate, which means the per-step error scales as O(h²). However, the energy error scales as O(h³), which makes it exceptional for Hamiltonian systems where long-term stability is critical.

The algorithm has three phases per step:
1. Partial momentum update (kick)
2. Full position update (drift)
3. Partial momentum update (kick)

The DepthMuun implementation adds friction and geometric constraints implicitly for stable Conformal Symplectic integration:

```markdown
v_{t+1/2} = \frac{v_t + \frac{h}{2} \cdot (F - \Gamma(x_t, v_t))}{1 + \frac{h}{2} \mu}
```

This implicit fractional form prevents gradient explosion during deep recursive calls and guarantees phase volume contraction.

Configurable parameters:
- `dt`: timestep (base value: 0.4)
- `friction_scale`: friction coefficient
- `substeps`: iterations per token step

### Yoshida

A fourth-order integrator that uses a sequence of 10 steps with optimized coefficients. The error scales as O(h⁵), significantly better than Leapfrog for the same timestep.

Better accuracy has a cost: more operations per step and higher memory usage to store intermediate states.

Useful when:
- High trajectory precision is required
- The timestep must be large
- Computational cost is not critical

Not recommended for production due to cost, but useful for benchmarking and validation.

### Forest-Ruth

Another fourth-order symplectic integrator, an alternative to Yoshida with a different structure. Less common in the literature but implemented for completeness.

The characteristics are similar to Yoshida: O(h⁵) error but more operations. The choice between Yoshida and Forest-Ruth is mostly a matter of preference.

### Omelyan

A second-order symplectic integrator optimized to minimize phase error. It combines Leapfrog simplicity with coefficients that reduce energy drift.

Useful when you need Leapfrog with better energy preservation, especially for long simulations.

## Runge-Kutta Integrators

Runge-Kutta integrators are generic methods for ordinary differential equations. They do not preserve symplectic structure but can be more accurate for non-Hamiltonian problems or when high precision is needed.

### Euler

The simplest integrator: y(t+h) = y(t) + h * f(y,t).

It does not preserve energy and diverges quickly. Included only for demonstration and testing, not for production use.

### Heun (Improved Euler)

Second-order method with predictor-corrector structure. More stable than Euler but less than Leapfrog for Hamiltonian systems.

Useful when friction dominates and symplectic structure is less important. This is used as the standard fallback integrator when computational speed is prioritized over strict topological preservation.

### Runge-Kutta 4 (RK4)

Standard fourth-order method. Accurate but expensive and does not preserve symplectic structure.

Useful for problems where accuracy is critical and the timestep is small.

## Comparison Table

| Integrator | Order | Symplectic Preservation | Relative Cost | Recommended Use |
|------------|-------|-------------------------|---------------|-----------------|
| Leapfrog   | 2     | Yes                     | 1.0x          | Generative tasks|
| Yoshida    | 4     | Yes                     | ~3.0x         | Default choice  |
| Forest-Ruth| 4     | Yes                     | ~3.0x         | High precision  |
| Omelyan    | 2     | Yes                     | ~1.2x         | Long simulations|
| Heun       | 2     | No                      | ~1.2x         | Speed-focused   |
| RK4        | 4     | No                      | ~4.0x         | Benchmarking    |

## Integrator Selection

For most tasks, Heun or Leapfrog is the right choice. They balance accuracy, stability, and efficiency.

Use Yoshida or Forest-Ruth when:
- The timestep must be large
- Trajectory precision is critical
- Computational cost is acceptable

Use Heun when:
- Default integrator is preferred
- Smooth transitions are needed

Avoid Euler in production. Its instability can cause divergence.

## Default Configuration

The system architecture defaults to `Yoshida` as the primary integrator to ensure maximum precision across heavily curved geometric manifolds, with the following standard configurations:
- `dt` = 0.4
- `curvature_clamp` = 3.0

This baseline config guarantees numerical stability even when facing high velocity impulses.

## Usage Example

```python
from gfn.core.config.manifold_configuration import ManifoldConfig, PhysicsConfig

# Using Heun integrator (fast fallback)
config_heun = ManifoldConfig(
    vocab_size=1000,
    dim=256,
    depth=4,
    integrator='heun',
    physics_config=PhysicsConfig(
        stability={'base_dt': 0.4}
    )
)

# Using Leapfrog integrator (symplectic operations)
config_leapfrog = ManifoldConfig(
    vocab_size=1000,
    dim=256,
    depth=4,
    integrator='leapfrog',
    physics_config=PhysicsConfig(
        stability={'base_dt': 0.4}
    )
)

# Using RK4 for non-symplectic high precision
config_rk4 = ManifoldConfig(
    vocab_size=1000,
    dim=256,
    depth=4,
    integrator='rk4',
    physics_config=PhysicsConfig(
        stability={'base_dt': 0.4}
    )
)
```

## Common Errors

If the loss diverges:
- Reduce base_dt in stability configuration
- Increase friction_scale
- Consider using a lower timestep

If the loss converges too slowly:
- Increase base_dt (up to 0.5)
- Reduce friction_scale
- Verify the integrator is appropriate for your use case

If you observe persistent oscillations:
- Friction may be too low
- Consider using Heun instead of Leapfrog
- Reduce impulse_scale in embedding configuration

---

**DepthMuun (GFN v2)**
