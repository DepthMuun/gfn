# GSSM Architecture Overview

The Geodesic State Space Model (G-SSM) is a physics-informed neural architecture that reformulates sequence modeling through differential geometry and Hamiltonian mechanics.

## Core Design Principles

1. **O(1) Inference Memory**: State evolves through a fixed-depth stack regardless of sequence length
2. **Symplectic Integration**: Energy-preserving numerical methods for stable dynamics
3. **Riemannian Geometry**: State evolution occurs on curved manifolds (torus, sphere, etc.)
4. **Hamiltonian Dynamics**: Position (x) and momentum (v) evolve according to physical laws

---

## Architecture Components

### 1. ManifoldModel (Entry Point)

```
gfn.create('gssm', ...) → ManifoldModel
```

The main model class that orchestrates:
- Embedding layer (token → manifold state)
- Stack of ManifoldLayers (depth layers)
- Readout layer (manifold state → logits)

### 2. ManifoldLayer

Each layer contains:
- **Integrator**: Numerical solver for ODEs (default: Leapfrog)
- **PhysicsEngine**: Computes accelerations from geometry
- **Geometry**: Christoffel symbols, metric tensor, friction
- **Mixer**: Head mixing for ensemble representations
- **Dynamics**: Routing mechanism (direct/residual/gated)

### 3. Physics Engine

Computes net acceleration:
```
dv/dt = -Γ(x,v) + F_ext + F_friction + F_ghost
```

Where:
- Γ(x,v): Christoffel symbols (geometric force)
- F_ext: External input force
- F_friction: Velocity-dependent damping
- F_ghost: Hysteresis ghost force

### 4. Integrator

Solves the Hamiltonian system:
- `LeapfrogIntegrator` (default): 2nd order symplectic
- `YoshidaIntegrator`: 4th order symplectic
- `VerletIntegrator`: 2nd order symplectic
- `RK4`: 4th order (non-symplectic)

---

## Data Flow

```
Input Tokens (B, S)
    ↓
Embedding (FunctionalEmbedding)
    ↓
x0, v0 (initial state)
    ↓
For each layer i in depth:
    ├─ PhysicsEngine.compute_acceleration(x, v, force)
    ├─ Integrator.step(x, v, force, dt) → x', v'
    ├─ Mixer(x', v') → mixed state
    └─ Dynamics(mixed state) → x_next, v_next
    ↓
Readout (manifold state → logits)
    ↓
Output Logits (B, S, V)
```

---

## Key Hyperparameters

### Model Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vocab_size` | 128 | >0 | Vocabulary size |
| `dim` | 64 | 32-512 | Total dimension |
| `heads` | 4 | 1-16 | Number of heads |
| `depth` | 4 | 1-16 | Number of layers |
| `initial_spread` | 0.0 | 0.0-1.0 | Initial state variance |

### Physics Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `base_dt` | 0.1 | 0.01-0.5 | Integration time step |
| `integrator_type` | leapfrog | - | Integration method |
| `friction` | 0.01 | 0.0-1.0 | Base friction coefficient |
| `topology_type` | torus | torus/euclidean | Manifold type |

---

## Topology Types

### Torus (Recommended)

- Bounded position space: [-π, π]
- Wraps around (periodic boundary)
- More stable dynamics
- Condition number: ~786 (tested)
- Position change: ~0.7 (stable)

### Euclidean

- Unbounded position space
- Can explode in magnitude
- Condition number: ~648
- Position change: ~45.7 (unstable)

**Recommendation**: Use `torus` for training.

---

## Integrator Comparison

Based on diagnostic tests:

| Integrator | x_norm | v_norm | Stability |
|------------|--------|--------|-----------|
| **leapfrog** | 14.3 | 37.7 | ✅ Most stable |
| yoshida | 19.8 | 99.0 | ⚠️ Higher energy |
| heun | 19.8 | 99.0 | ⚠️ Higher energy |

**Recommendation**: Keep `leapfrog` (default).

---

## Time Step (dt) Sensitivity

Based on diagnostic tests:

| dt | v_norm | Stability |
|----|--------|-----------|
| 0.01 | 10.6 | ✅ Conservative |
| 0.05 | 52.3 | ✅ Good |
| **0.1** | 98.9 | ✅ Default |
| 0.2 | 123.8 | ⚠️ Higher |
| 0.5 | 137.6 | ⚠️ May be unstable |

**Recommendation**: Use `dt=0.05-0.1` for training.

---

## Known Issues & Solutions

### 1. Rank Deficiency in First Layer

**Symptom**: Condition number ~17M in layer 0

**Cause**: Embedding projection bottleneck

**Solution**:
- Increase `initial_spread` to 0.1-0.5
- Add layer normalization after embedding

### 2. Vanishing Gradients

**Symptom**: min_gradient_norm ~1e-08

**Solution**:
- Use `initial_spread=0.1-0.5`
- Add residual connections

### 3. Numerical Instability

**Symptom**: NaN in forward pass

**Solution**:
- Reduce `base_dt` to 0.05
- Use `leapfrog` integrator
- Enable trace normalization

---

## Best Practices

1. **Initialization**: Use `initial_spread=0.1` for better gradient flow
2. **Training**: Use `base_dt=0.05-0.1` and `leapfrog` integrator
3. **Topology**: Use `torus` for stable training
4. **Monitoring**: Track velocity norms - should be < 100 for stability

---

## File Structure

```
gfn/realizations/gssm/
├── models/
│   ├── manifold.py      # Main model
│   ├── manifold_layer.py # Per-layer logic
│   ├── base.py          # Core evolution
│   └── factory.py       # Model creation
├── physics/
│   ├── engine.py        # Physics computation
│   ├── integrators/     # Numerical solvers
│   │   ├── symplectic/  # Energy-preserving
│   │   └── runge_kutta/ # Standard methods
│   └── dynamics/        # Dynamics routing
├── geometry/
│   ├── torus.py         # Toroidal manifold
│   ├── low_rank.py      # Low-rank approximation
│   └── factory.py       # Geometry creation
└── config/
    ├── schema.py        # Configuration classes
    └── defaults.py      # Default values
```

---

## Quick Start

```python
import gfn

# Basic model
model = gfn.create(
    'gssm',
    vocab_size=1000,
    dim=64,
    heads=4,
    depth=4
)

# Optimized configuration (based on diagnostics)
model = gfn.create(
    'gssm',
    vocab_size=1000,
    dim=64,
    heads=4,
    depth=4,
    initial_spread=0.1,      # Better gradient flow
    physics={
        'stability': {
            'base_dt': 0.05,     # More stable
            'integrator_type': 'leapfrog'
        },
        'topology': {
            'type': 'torus'      # Stable manifold
        }
    }
)
```

---

*Generated from diagnostic test results - 2026-04-02*
