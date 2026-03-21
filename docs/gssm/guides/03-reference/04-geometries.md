# Geometry Reference

## Overview

Geometries define the metric and connection (Christoffel symbols) of the manifold on which the system operates. The choice of geometry affects what structures the model can learn and how the state evolves.

## LowRankRiemannianGeometry

The system's main geometry, using low-rank factorization for efficiency.

### Characteristics

- Configurable rank to balance accuracy/speed
- Optional trace normalization
- Configurable curvature clamping

### Parameters

```python
# Setup the config first
physics_config = PhysicsConfig(
    stability={'curvature_clamp': 3.0}
)

LowRankRiemannianGeometry(
    dim=512,
    physics_config=physics_config
)
```

### Typical Usage

```python
geometry = LowRankRiemannianGeometry(
    dim=model.dim,
    physics_config=model.physics_config
)
christoffel = geometry(velocity, position)
```

### Accuracy vs Speed

| Rank | Relative Accuracy | Relative Speed |
|------|-------------------|-------------------|
| 16 | 95% | 4x |
| 32 | 98% | 2x |
| 64 | 99.5% | 1x |
| 128 | 99.9% | 0.5x |

## Hyperbolic Geometry

Implements geometry of hyperbolic space H^n.

### Characteristics

- Constant negative curvature
- Appropriate for hierarchical structures
- Analytical metric (no neural network required)

### Metric

The hyperbolic space metric approximation forces paths outwards:

$g = \frac{4}{(1 - \|x\|^2)^2} I$

Where $\|x\| < 1$. This metric "grows" near the edge of the unit disk.

### Parameters

```python
HyperbolicGeometry(
    dim=512,
    physics_config=physics_config
)
```

### Use Cases

- Syntax trees
- Taxonomies
- Hierarchical graphs
- Data with inclusion structure

### Limitations

- Requires embeddings to be inside the unit disk
- The exponential map can overflow for points near the edge
- Not appropriate for data without hierarchical structure

## ToroidalRiemannianGeometry

Implements periodic boundaries for constrained topological paths.

### Characteristics

- Periodic dimensions (angles)
- Non-periodic dimensions (radii)
- Curvature that changes sign

### Metric

For a point (Î¸â‚, Î¸â‚‚, râ‚, râ‚‚) in toroidal coordinates:

g_ij = diag(1, 1, râ‚Â², râ‚‚Â²) for coordinates (Î¸â‚, Î¸â‚‚, râ‚, râ‚‚)

### Parameters

```python
ToroidalGeometry(
    dim=512,
    n_periodic=2,        # Periodic dimensions
    major_radius=2.0,    # Major torus radius
    minor_radius=1.0     # Minor radius
)
```

### Use Cases

- Cyclic time series
- Angular coordinates
- Data with periodic structure
- Problems with rotational symmetries

### Limitations

- Only 2 dimensions are periodic by default
- Requires preprocessing data into toroidal coordinates
- The metric near the origin is nearly flat

## Analytical Geometries

Uses predefined analytical formulas for strict boundary constraints without neural generation.

### Available Metrics

**EuclideanGeometry:**

```python
EuclideanGeometry(dim=512)
```

**SphericalGeometry:**

```python
SphericalGeometry(dim=512)
```

**HyperbolicGeometry:**

```python
HyperbolicGeometry(dim=512)
```

## AdaptiveRiemannianGeometry

Adjusts the metric based on input data.

### Characteristics

- Metric that changes with the state
- Configurable plasticity
- Curvature that emerges from data

### Parameters

```python
AdaptiveRiemannianGeometry(
    dim=512,
    physics_config=physics_config
)
```

### Dynamics

Curvature adapts smoothly according to:

$dK/dt = \text{plasticity} \cdot (\text{adaptation\_signal} - K)$

The adaptation signal comes from the "energy" of the input data.

### Use Cases

- Data with unknown structure
- Problems where geometry must emerge
- Experimentation with learned geometries

### Limitations

- More parameters to tune
- Can overfit geometry to noise
- Slower convergence

## ReactiveRiemannianGeometry

Dynamic response of the metric to extreme system instability.

### Characteristics

- Curvature that responds to "forces" to brake runaway trajectories.
- Implements Active Inference triggers based on kinetic energy limits.

### Parameters

```python
ReactiveRiemannianGeometry(
    dim=512,
    physics_config=physics_config
)
```

### Dynamics

When curvature exceeds the configured capacity threshold from `physics_config`, a stabilizing force is automatically applied.

## Geometry Comparison

| Geometry | Trainable | Cost | Use Cases |
|-----------|------------|-------|--------------|
| LowRankRiemannian | Yes | Medium | Default Standard |
| Hyperbolic | No | Low | Hierarchies |
| ToroidalRiemannian| No | Low | Cyclic logic |
| AdaptiveRiemannian| Yes | High | Emergent structure |
| ReactiveRiemannian| Yes | Medium | Stability enforcement |

## Geometry Selection

For most tasks, `LowRankRiemannianGeometry` is the required baseline.

Use `HyperbolicGeometry` if:
- Data has clear hierarchical structure
- The vocabulary has inclusion relationships

Use `ToroidalRiemannianGeometry` if:
- Data includes angular constraints
- Target outputs depend on modulo arithmetic (XOR parity)

## Default Configuration

```python
geometry = LowRankRiemannianGeometry(
    dim=512,
    physics_config=physics_config
)
```

This ensures a linear parameter scaling suitable for sequences over 100,000 steps deep.

---

**DepthMuun (GFN v2)**
