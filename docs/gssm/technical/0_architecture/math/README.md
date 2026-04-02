# GSSM Mathematical Foundations

Complete mathematical documentation for the Geodesic State Space Model.

---

## Directory Structure

### Core Physics

| File | Topic | What It Explains |
|------|-------|------------------|
| `01_physics_engine.md` | Physics Engine | What is acceleration, how forces combine, equations |
| `03_geometry.md` | Geometries Overview | Christoffel symbols, manifold curvature (general) |
| `04_hysteresis.md` | Hysteresis | Memory mechanism, ghost forces |
| `05_singularities.md` | Singularities | Numerical stability, damping |
| `06_stochasticity.md` | Stochasticity | Random forces, Brownian/OU processes |
| `07_curiosity.md` | Curiosity | Exploration force |
| `08_dynamics.md` | Dynamics | State update routing (direct/residual/gated) |

### Integrators (Individual Files)

See `integrators/` subdirectory for detailed explanations of each integrator:

| Integrator | File | Order | Type |
|------------|------|-------|------|
| Leapfrog | `integrators/leapfrog.md` | 2nd | **Symplectic (default)** |
| Yoshida | `integrators/yoshida.md` | 4th | Symplectic |
| Verlet | `integrators/verlet.md` | 2nd | Symplectic |
| Forest-Ruth | `integrators/forest_ruth.md` | 4th | Symplectic |
| Omelyan | `integrators/omelyan.md` | 2nd | Symplectic |
| RK4 | `integrators/rk4.md` | 4th | Runge-Kutta |
| Heun | `integrators/heun.md` | 2nd | Runge-Kutta |

### Data Flow

| File | Topic | What It Explains |
|------|-------|------------------|
| `forward_pass_conceptual.md` | Forward Pass | How input becomes output (3 phases) |
| `backward_pass_conceptual.md` | Backward Pass | How gradients flow through all components |

### Components

See `components/` subdirectory:

| Component | File | Purpose |
|-----------|------|---------|
| Mixer | `components/mixer.md` | Combines information across heads |
| Embedding | `components/embedding.md` | Token → force mapping |
| Readout | `components/readout.md` | State → logits mapping |
| Normalization | `components/normalization.md` | Position/velocity bounds |

### Plugins

See `plugins/` subdirectory:

| Plugin | File | Purpose |
|--------|------|---------|
| Dynamic Time | `plugins/dynamic_time.md` | Per-head adaptive time steps |
| Fractal | `plugins/fractal.md` | Micro-manifold refinement |

### System Architecture

See `system/` subdirectory:

| Component | File | Purpose |
|-----------|------|---------|
| Hooks | `system/hooks.md` | Lifecycle injection points |
| Factory | `system/factory.md` | Model construction |

### Training

See `training/` subdirectory:

| Component | File | Purpose |
|-----------|------|---------|
| Losses | `training/losses.md` | Physics-informed loss functions |
| Optimizers | `training/optimizers.md` | Riemannian Adam, dual-group optimization |

### Geometries

See `geometry/` subdirectory:

| Geometry | File | Curvature | Bounded |
|----------|------|-----------|---------|
| Torus | `geometry/torus.md` | Variable | Yes |
| Euclidean | `geometry/euclidean.md` | Flat | No |
| Low-Rank | `geometry/low_rank.md` | Approximate | Yes |

---

## Quick Reference

### The Fundamental Equation

**Acceleration** (Physics Engine):

$$a = -\Gamma(x,v) + F_{ext} - \mu v + F_{ghost} + F_{stochastic} + F_{curiosity}$$

Where:
- $-\Gamma(x,v)$ = Christoffel force (geometry)
- $F_{ext}$ = External force (embedding)
- $-\mu v$ = Friction
- $F_{ghost}$ = Hysteresis ghost force
- $F_{stochastic}$ = Brownian/OU noise
- $F_{curiosity}$ = Repulsion from batch center

### State Evolution (Leapfrog)

**Kick** (half-step velocity):
$$v_{n+1/2} = v_n + \frac{\Delta t}{2} \cdot a(x_n, v_n)$$

**Drift** (full-step position):
$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$

**Kick** (complete velocity):
$$v_{n+1} = v_{n+1/2} + \frac{\Delta t}{2} \cdot a(x_{n+1}, v_{n+1/2})$$

### Dynamics Routing

**Direct**:
$$x_{next} = x_{proposal}$$

**Residual**:
$$x_{next} = x + \sigma(s) \cdot (x_{proposal} - x)$$

**Gated**:
$$x_{next} = g \cdot x_{proposal} + (1-g) \cdot x$$

---

## Complete File List (27 Files)

```
math/
├── README.md                              # This file
├── 01_physics_engine.md                   # Acceleration components
├── 02_integrators.md                        # Integrators overview
├── 03_geometry.md                           # Geometries overview
├── 04_hysteresis.md                         # Memory mechanism
├── 05_singularities.md                      # Singularity handling
├── 06_stochasticity.md                      # Random forces
├── 07_curiosity.md                          # Exploration force
├── 08_dynamics.md                           # State routing
├── 09_forward_pass.md                       # Detailed forward pass
├── 10_backward_pass.md                      # Detailed backward pass
├── 11_integrators_detailed.md               # All integrators code
├── forward_pass_conceptual.md              # Conceptual forward
├── backward_pass_conceptual.md             # Conceptual backward
├── integrators/
│   ├── README.md                           # Integrator selection
│   ├── leapfrog.md                         # Default integrator
│   ├── yoshida.md                          # 4th order symplectic
│   ├── verlet.md                           # Velocity Verlet
│   ├── forest_ruth.md                      # Alternative 4th order
│   ├── omelyan.md                          # Optimized 2nd order
│   ├── rk4.md                              # Runge-Kutta 4
│   └── heun.md                             # Improved Euler
├── components/
│   ├── mixer.md                            # Head mixing
│   ├── embedding.md                        # Token → force
│   ├── readout.md                          # State → logits
│   └── normalization.md                    # State bounds
├── plugins/
│   ├── dynamic_time.md                     # Adaptive dt
│   └── fractal.md                          # Micro-manifold
├── system/
│   ├── hooks.md                            # Lifecycle hooks
│   └── factory.md                          # Model builder
├── training/
│   ├── losses.md                           # Physics losses
│   └── optimizers.md                       # Riemannian optimizers
└── geometry/
    ├── README.md                           # Geometry selection
    ├── torus.md                            # Default geometry
    ├── euclidean.md                        # Flat space
    └── low_rank.md                         # Efficient approx
```

---

## Reading Guide

### For Understanding the Model

1. **Start**: `forward_pass_conceptual.md` - What happens during inference
2. **Then**: `01_physics_engine.md` - How acceleration is computed
3. **Then**: `integrators/leapfrog.md` - How state evolves
4. **Then**: `03_geometry.md` - Where curvature comes from
5. **Finally**: `backward_pass_conceptual.md` - How learning works

### For Component Details

| Interest | Read |
|----------|------|
| How heads mix | `components/mixer.md` |
| Token embedding | `components/embedding.md` |
| Output generation | `components/readout.md` |
| Adaptive time | `plugins/dynamic_time.md` |
| Training losses | `training/losses.md` |
| Optimization | `training/optimizers.md` |

### For Implementation

- **API Reference**: `../guides/03-reference/02-api-classes.md`
- **Architecture Overview**: `../02_code_analysis.md`
- **Troubleshooting**: `../guides/04-guides/03-problem-solving.md`

---

## Key Concepts

### What is a Symplectic Integrator?

A symplectic integrator preserves the symplectic 2-form in phase space:

$$\omega = dp \wedge dq$$

**Benefits**:
- Energy oscillates but doesn't drift
- Good for long-term stability
- Required for Hamiltonian systems

**Symplectic methods**: Leapfrog, Yoshida, Verlet, Forest-Ruth, Omelyan

**Non-symplectic**: RK4, Heun (energy drifts over time)

### What is the Manifold?

The manifold is the space where the state $(x, v)$ lives:

- **Torus** ($S^1 \times ... \times S^1$): Periodic, bounded, stable
- **Euclidean** ($\mathbb{R}^n$): Unbounded, can explode

Position $x$ represents "where" on the manifold.
Velocity $v$ represents "how fast" along the manifold.

### What is Christoffel?

Christoffel symbols $\Gamma^k_{ij}$ represent manifold curvature:

$$\text{Geodesic force} = -\Gamma(x, v)$$

They push the state along the "straightest possible line" on curved space.

---

## Cross-References

- **Complete API**: `../guides/03-reference/02-api-classes.md`
- **Architecture Overview**: `../02_code_analysis.md`
- **Troubleshooting**: `../guides/04-guides/03-problem-solving.md`

---

*Last Updated: 2026-04-02*
*Version: GSSM v2.7.2*
*Total Files: 27*
