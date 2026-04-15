# GSSM API Reference

Complete reference for all GSSM configuration parameters, classes, and usage.

## 1. Quick Start

### Basic Model Creation

```python
import gfn

# Minimal model
model = gfn.create('gssm', vocab_size=1000)

# Full configuration
model = gfn.create(
    'gssm',
    vocab_size=1000,
    dim=64,
    heads=4,
    depth=4,
    initial_spread=0.1,
    physics={
        'stability': {
            'base_dt': 0.05,
            'integrator_type': 'leapfrog',
            'friction': 0.01
        },
        'topology': {
            'type': 'torus',
            'R': 2.0,
            'r': 1.0
        }
    }
)
```

---

## 2. Model Configuration Parameters

### Top-Level Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | **Required** | Vocabulary size |
| `dim` | int | 64 | Total model dimension |
| `heads` | int | 4 | Number of attention heads |
| `depth` | int | 4 | Number of manifold layers |
| `rank` | int | 16 | Low-rank dimension for mixing |
| `holographic` | bool | False | Use holographic embeddings |
| `initial_spread` | float | 0.1 | Initial state variance |
| `n_trajectories` | int | 1 | Number of trajectories |

---

## 3. Physics Configuration

### 3.1 TopologyConfig

Controls the manifold topology.

```python
physics={
    'topology': {
        'type': 'torus',
        'R': 2.0,
        'r': 1.0,
        'learnable_r': True
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | 'torus' | Topology type |
| `R` | float | 2.0 | Major radius (torus) |
| `r` | float | 1.0 | Minor radius (torus) |
| `learnable_r` | bool | True | Make r learnable |

**Available topologies**: `torus`, `euclidean`, `sphere`, `hyperbolic`, `low_rank`, `reactive`, `adaptive`, `holographic`, `hierarchical`

### 3.2 StabilityConfig

Controls numerical stability and integration.

```python
physics={
    'stability': {
        'base_dt': 0.1,
        'adaptive': True,
        'dt_min': 0.001,
        'dt_max': 1.0,
        'enable_trace_normalization': True,
        'wrap_x': True,
        'friction': 0.01,
        'velocity_friction_scale': 0.0,
        'velocity_saturation': 0.0,
        'curvature_clamp': 100000.0,
        'friction_mode': 'static',
        'integrator_type': 'leapfrog',
        'toroidal_curvature_scale': 0.01
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dt` | float | 0.1 | Integration time step |
| `adaptive` | bool | True | Enable adaptive dt |
| `dt_min` | float | 0.001 | Minimum dt |
| `dt_max` | float | 1.0 | Maximum dt |
| `enable_trace_normalization` | bool | True | Enable metric normalization |
| `wrap_x` | bool | True | Wrap to manifold bounds |
| `friction` | float | 0.01 | Base friction coefficient |
| `velocity_friction_scale` | float | 0.0 | Velocity-dependent friction |
| `velocity_saturation` | float | 0.0 | Max velocity (0=disabled) |
| `curvature_clamp` | float | 1e5 | Max Christoffel magnitude |
| `friction_mode` | str | 'static' | 'static' or 'lif' |
| `integrator_type` | str | 'leapfrog' | Integration method |
| `toroidal_curvature_scale` | float | 0.01 | Torus curvature multiplier |

### 3.3 Integrator Types

| Type | Order | Symplectic | Description |
|------|-------|------------|-------------|
| `leapfrog` | 2nd | ✅ | Default, most stable |
| `verlet` | 2nd | ✅ | Velocity Verlet |
| `yoshida` | 4th | ✅ | High precision |
| `forest_ruth` | 4th | ✅ | Specialized Hamiltonian |
| `rk4` | 4th | ❌ | Classic Runge-Kutta |
| `heun` | 2nd | ❌ | Improved Euler |

**Recommendation**: Use `leapfrog` for training stability.

### 3.4 DynamicsConfig

Controls state update mechanism.

```python
physics={
    'dynamics': {
        'type': 'direct'
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | 'direct' | 'direct', 'residual', 'gated' |

---

## 4. Advanced Configuration

### 4.1 ActiveInferenceConfig

Enables adaptive geometry and dynamics.

```python
physics={
    'active_inference': {
        'enabled': False,
        'holographic_geometry': False,
        'thermodynamic_geometry': False,
        'plasticity': 0.05,
        'dynamic_time': {
            'enabled': False,
            'type': 'riemannian'
        },
        'reactive_curvature': {
            'enabled': False,
            'plasticity': 0.0
        },
        'geodesic_lensing': {
            'enabled': False
        },
        'stochasticity': {
            'enabled': False,
            'type': 'brownian',
            'sigma': 0.01,
            'theta': 0.15,
            'mu': 0.0
        },
        'curiosity': {
            'enabled': False,
            'strength': 0.1,
            'decay': 0.99
        }
    }
}
```

### 4.2 HysteresisConfig

Enables hysteresis-based memory.

```python
physics={
    'hysteresis': {
        'enabled': False,
        'ghost_force': True,
        'hyst_decay': 0.1,
        'hyst_update_w': 1.0,
        'hyst_update_b': 0.0,
        'hyst_readout_w': 1.0,
        'hyst_readout_b': 0.0
    }
}
```

### 4.3 FractalConfig

Enables fractal step refinement.

```python
physics={
    'fractal': {
        'enabled': False,
        'threshold': 0.5,
        'alpha': 0.2
    }
}
```

### 4.4 SingularityConfig

Handles geometric singularities.

```python
physics={
    'singularities': {
        'enabled': False,
        'epsilon': 1e-8,
        'strength': 0.1,
        'threshold': 0.0001
    }
}
```

### 4.5 EmbeddingConfig

Controls token embedding.

```python
physics={
    'embedding': {
        'type': 'standard',
        'mode': 'linear',
        'coord_dim': 16,
        'impulse_scale': 1.0,
        'omega_0': 30.0
    }
}
```

---

## 5. Complete Configuration Example

```python
import gfn

# Complete model with all options
model = gfn.create(
    'gssm',
    vocab_size=1000,
    dim=64,
    heads=4,
    depth=4,
    rank=16,
    holographic=False,
    initial_spread=0.1,
    n_trajectories=1,
    physics={
        'topology': {
            'type': 'torus',
            'R': 2.0,
            'r': 1.0,
            'learnable_r': True
        },
        'stability': {
            'base_dt': 0.05,
            'adaptive': True,
            'dt_min': 0.001,
            'dt_max': 1.0,
            'enable_trace_normalization': True,
            'wrap_x': True,
            'friction': 0.01,
            'velocity_friction_scale': 0.0,
            'velocity_saturation': 0.0,
            'curvature_clamp': 100000.0,
            'friction_mode': 'static',
            'integrator_type': 'leapfrog',
            'toroidal_curvature_scale': 0.01
        },
        'dynamics': {
            'type': 'direct'
        },
        'active_inference': {
            'enabled': False,
            'holographic_geometry': False,
            'thermodynamic_geometry': False,
            'plasticity': 0.05,
            'dynamic_time': {'enabled': False, 'type': 'riemannian'},
            'reactive_curvature': {'enabled': False, 'plasticity': 0.0},
            'geodesic_lensing': {'enabled': False},
            'stochasticity': {'enabled': False, 'type': 'brownian', 'sigma': 0.01, 'theta': 0.15, 'mu': 0.0},
            'curiosity': {'enabled': False, 'strength': 0.1, 'decay': 0.99}
        },
        'hysteresis': {
            'enabled': False,
            'ghost_force': True,
            'hyst_decay': 0.1,
            'hyst_update_w': 1.0,
            'hyst_update_b': 0.0,
            'hyst_readout_w': 1.0,
            'hyst_readout_b': 0.0
        },
        'fractal': {
            'enabled': False,
            'threshold': 0.5,
            'alpha': 0.2
        },
        'singularities': {
            'enabled': False,
            'epsilon': 1e-8,
            'strength': 0.1,
            'threshold': 0.0001
        },
        'embedding': {
            'type': 'standard',
            'mode': 'linear',
            'coord_dim': 16,
            'impulse_scale': 1.0,
            'omega_0': 30.0
        }
    }
)
```

---

## 6. Training

```python
import torch
import gfn

# Create model
model = gfn.create('gssm', vocab_size=1000, dim=64, depth=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    inputs, targets = batch
    
    logits, (xf, vf), info = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 1000), 
        targets.reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 7. Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_DT` | 0.1 | Default time step |
| `DEFAULT_FRICTION` | 0.01 | Default friction |
| `DEFAULT_PLASTICITY` | 0.05 | Default plasticity |
| `MAX_VELOCITY` | 10.0 | Velocity saturation |
| `CURVATURE_CLAMP` | 1e5 | Christoffel clamp |
| `SINGULARITY_THRESHOLD` | 1e-4 | Singularity detection |
| `EPSILON_STANDARD` | 1e-8 | Standard epsilon |

---

*Last Updated: 2026-04-02*
*Version: GSSM v2.7.2*
