# Version Comparison: v2.6.5 vs Development

## Summary

This document analyzes the differences between the stable v2.6.5 release and the current development version. The goal is to understand why the development version may behave differently and how to recover convergence if needed.

## Main Differences

### Overall Architecture

**v2.6.5** had a monolithic architecture where integrator logic, geometry, and loss were intertwined. The Manifold constructor accepted only basic parameters.

**Development** modularizes these components. The integrator is swappable, geometry can be changed, and loss terms are composable. The cost is greater configuration complexity.

### Model Constructor

**v2.6.5:**

```python
class Manifold(nn.Module):
    def __init__(self, vocab_size, dim, depth, rank, heads, integrator_type):
        # Only 6 parameters
```

**Development:**

```python
class Manifold(nn.Module):
    def __init__(self, vocab_size, dim, depth, rank, heads, integrator_type,
                 hyst_alpha=0.9, hyst_beta=0.1,
                 enable_trace_normalization=True,
                 velocity_friction_scale=0.02,
                 position_friction_scale=0.0,
                 plasticity_coef=0.02,
                 singularity_threshold=0.5,
                 # ... more parameters
                 ):
```

### Leapfrog Integrator

**v2.6.5** used an explicit update:

```python
v_half = curr_v + h * (force - gamma)
```

**Development** uses an implicit update:

```python
v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON)
```

The change improves stability but alters the system's physics.

### Hysteresis System

**v2.6.5** had no hysteresis.

**Development** adds a long-term memory system:

- HYSTERESIS_FORGET_GATE_INIT = 0.9
- HYSTERESIS_STATE_MOMENTUM = 0.95
- HYSTERESIS_GHOST_FORCE_SCALE = 0.1

Hysteresis lets the model "remember" previous states but adds state variables and complexity.

### Velocity-Dependent Friction

**v2.6.5** had constant friction.

**Development** adds friction that scales with velocity:

```python
velocity_friction = velocity_friction_scale * norm(p)
total_friction = base_friction + velocity_friction
```

This makes the system more stable for fast dynamics.

## Constants Analysis

### Physics Constants

| Constant | v2.6.5 | Development | Effect |
|-----------|--------|------------|--------|
| FRICTION_SCALE | 5.0 | 0.02 | 250x lower |
| DEFAULT_FRICTION | 0.05 | 0.002 | 25x lower |
| CURVATURE_CLAMP | 20.0 | 3.0 | 6.7x lower |

The pattern is clear: the development version is more permissive with manifold exploration.

### Optimizer Constants

| Constant | v2.6.5 | Development | Effect |
|-----------|--------|------------|--------|
| DEFAULT_LR | 1e-3 | 1e-4 | 10x lower |
| ADAM_BETA2 | 0.999 | 0.99 | More stable |
| DEFAULT_DT | 0.1 | 0.05 | 2x lower |

The optimizer is more conservative in the development version.

### Stability Constants

| Constant | v2.6.5 | Development | Effect |
|-----------|--------|------------|--------|
| EPSILON_STANDARD | 1e-6 | 1e-7 | 10x lower |
| EPSILON_STRONG | 1e-4 | 1e-7 | 1400x lower |

Smaller epsilons allow better precision but less protection against underflow.

## Differences in Loss Functions

### Hamiltonian Loss

**v2.6.5** used:

```python
h_loss = lambda_h * abs(H_final - H_initial)
```

**Development** uses:

```python
h_loss = lambda_h * torch.mean((H - H_initial)**2)
```

The current version penalizes squared deviation instead of absolute deviation.

### Geodesic Loss

**v2.6.5** had fixed regularization.

**Development** allows modes:

```python
# Standard mode
g_loss = lambda_g * geodesic_deviation(trajectory)

# Adaptive mode
g_loss = lambda_g * adaptive_geodesic_loss(trajectory)
```

### New Loss Terms

**Development** adds:

- **LAMBDA_N_DEFAULT**: Noether symmetry loss
- **LAMBDA_K_DEFAULT**: Kinetic energy penalty

## Differences in Geometry

### Metric

**v2.6.5** computed the metric in a single way.

**Development** adds options:

```python
# Trace normalization
if enable_trace_normalization:
    metric = metric / trace(metric) * dim
```

### Christoffel Symbols

**v2.6.5** computed Christoffel directly.

**Development** adds:

- Improved low-rank approximation
- Configurable curvature clamping
- Optional normalization

## Differences in the CUDA Backend

### Kernels

**v2.6.5** had separate kernels.

**Development** fuses operations:

- `leapfrog_fused`: Kick-Drift-Kick in one kernel
- `christoffel_fused`: Full computation in one kernel

Fusion reduces memory bandwidth and increases throughput.

### Backward Pass

**v2.6.5** backward had 7 outputs.

**Development** backward has 11 outputs to include hysteresis gradients.

### Parity Verification

**v2.6.5** had no parity tests.

**Development** includes automated tests that verify Python vs CUDA.

## Documentation Differences

**v2.6.5** had fragmented documentation:
- README.md básico
- Papers en docs/00_papers/
- Sin índice centralizado

**Development** reorganizes documentation:
- docs/00-INDICE.md with navigation
- Módulos temáticos (01-introduccion, 02-conceptos-core, etc.)
- Practical usage guides

## Convergence Hypotheses

### Why v2.6.5 Converged

1. **High friction (5.0)** forced rapid convergence
2. **High LR (1e-3)** accelerated optimization
3. **Few hyperparameters** meant fewer ways to fail
4. **Conservative defaults** had been empirically tuned

### Why Development May Not Converge

1. **Low friction (0.02)** allows infinite oscillation
2. **Low LR (1e-4)** converges more slowly
3. **More hyperparameters** = more ways to unbalance
4. **New loss terms** can compete with each other
5. **Physics changes** (implicit integrator) alter dynamics

## Recommendations to Recover Convergence

### Step 1: Stable Baseline

Start with a configuration that reliably converges:

```yaml
physics:
  friction_scale: 0.5
  dt: 0.05
  lambda_g: 0.0001

training:
  learning_rate: 5e-4
  adam_beta2: 0.999
```

### Step 2: Introduce New Features

Once it converges, add features gradually:

1. Add hysteresis (hyst_alpha=0.9)
2. Enable velocity friction (0.1)
3. Increase LEAPFROG_SUBSTEPS
4. Reduce friction_scale to 0.1

### Step 3: Fine Tuning

With a functional baseline, optimize:

- Lower LR gradually to 1e-4
- Adjust lambda_g by task
- Experiment with alternative integrators

## Equivalence Table

| v2.6.5 | Development | Notes |
|--------|------------|-------|
| FRICTION_SCALE=5.0 | FRICTION_SCALE=0.5 | 10x difference |
| DEFAULT_LR=1e-3 | DEFAULT_LR=5e-4 | 2x difference |
| DEFAULT_DT=0.1 | DEFAULT_DT=0.08 | 1.25x difference |
| LAMBDA_H=0.01 | LAMBDA_H=0.0 | Disabled |
| LAMBDA_G=0.001 | LAMBDA_G=0.00005 | 20x difference |

## Conclusions

The differences between versions are significant but manageable. The development version is more flexible but requires more tuning. The key is to start with conservative values and add complexity gradually.

The v2.6.5 version converged not because it was "better" but because it was more restrictive. The development version can match or exceed that performance with the right tuning.

## References

- Logic and math audit: AUDIT_MATH_LOGIC_CORE_2026_02_07.md
- Constant reversion viability analysis: 02-viabilidad-reversion-constantes.md
- Constants documentation: 03-referencia/01-constantes.md

---

**Manifold Labs (Joaquín Stürtz)**
