# Geometry Reference

## Overview

Geometries define the metric and connection (Christoffel symbols) of the manifold on which the system operates. The choice of geometry affects what structures the model can learn and how the state evolves.

## ChristoffelLowRank

The system's main geometry, using low-rank factorization for efficiency.

### Characteristics

- Configurable rank to balance accuracy/speed
- Optional trace normalization
- Configurable curvature clamping

### Parameters

```python
ChristoffelLowRank(
    dim=512,              # Manifold dimension
    rank=64,              # Factorization rank
    curvature_clamp=3.0,  # Curvature limit
    enable_trace_normalization=True
)
```

### Typical Usage

```python
geometry = ChristoffelLowRank(
    dim=model.dim,
    rank=model.rank,
    curvature_clamp=3.0
)
christoffel = geometry(input_embedding)
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

La métrica del espacio hiperbólico es:

g = (4 / (1 - ||x||²)²) * I

Where ||x|| < 1. This metric "grows" near the edge of the unit disk.

### Parameters

```python
HyperbolicGeometry(
    dim=512,
    curvature=-1.0,  # Negative curvature (hyperbolic)
    radius=1.0       # Space radius
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

## Toroidal Geometry

Implements geometry on a torus T² × R^{n-2}.

### Characteristics

- Periodic dimensions (angles)
- Non-periodic dimensions (radii)
- Curvature that changes sign

### Metric

For a point (θ₁, θ₂, r₁, r₂) in toroidal coordinates:

g_ij = diag(1, 1, r₁², r₂²) for coordinates (θ₁, θ₂, r₁, r₂)

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

## Analytical Geometry

Uses predefined analytical formulas for known metrics.

### Characteristics

- No trainable parameters
- Closed-form metric (no network required)
- Useful for benchmarking and testing

### Available Metrics

**Euclidean Metric:**

```python
AnalyticalGeometry(
    metric_type="euclidean"
)
```

**Spherical Metric:**

```python
AnalyticalGeometry(
    metric_type="spherical",
    radius=1.0
)
```

**Poincaré Metric:**

```python
AnalyticalGeometry(
    metric_type="poincare",
    curvature=-1.0
)
```

## Adaptive Geometry

Adjusts the metric based on input data.

### Characteristics

- Metric that changes with the state
- Configurable plasticity
- Curvature that emerges from data

### Parameters

```python
AdaptiveGeometry(
    dim=512,
    plasticity=0.02,           # Adaptation speed
    threshold=0.5,             # Activation threshold
    curvature_lr=0.01          # Curvature learning rate
)
```

### Dynamics

La curvatura evoluciona según:

dK/dt = plasticity * (adaptation_signal - K)

The adaptation signal comes from the "energy" of the input data.

### Use Cases

- Data with unknown structure
- Problems where geometry must emerge
- Experimentation with learned geometries

### Limitations

- More parameters to tune
- Can overfit geometry to noise
- Slower convergence

## Hierarchical Geometry

Combines multiple levels of detail in the metric.

### Characteristics

- Multi-scale resolution
- High-frequency detail
- Low-frequency structure

### Parameters

```python
HierarchicalGeometry(
    dim=512,
    n_levels=3,          # Number of levels
    base_rank=16,        # Base level rank
    detail_factor=1.5    # Detail factor per level
)
```

### Use Cases

- Data with multi-scale structure
- Fractals
- Images and signals with detail

## Reactive Geometry

Dynamic response of the metric to perturbations.

### Characteristics

- Curvature that responds to "forces"
- "Black hole" effect for stabilization
- Curvature memory

### Parameters

```python
ReactiveGeometry(
    dim=512,
    black_hole_strength=1.5,  # Stabilization strength
    reactive_lr=0.01,         # Response speed
    max_adjustment=0.1        # Maximum adjustment per step
)
```

### Dynamics

When curvature exceeds a threshold, a stabilizing force is applied:

adjustment = -black_hole_strength * (K - threshold)

This prevents singularities and keeps the metric well conditioned.

## Geometry Comparison

| Geometry | Trainable | Cost | Use Cases |
|-----------|------------|-------|--------------|
| LowRank | Yes | Medium | General |
| Hyperbolic | No | Low | Hierarchies |
| Toroidal | No | Low | Cyclic |
| Adaptive | Yes | High | Emergent structure |
| Hierarchical | Yes | High | Multi-scale |
| Reactive | Yes | Medium | Stability |

## Geometry Selection

For most tasks, ChristoffelLowRank is the right choice.

Use Hyperbolic if:
- Data has clear hierarchical structure
- The vocabulary has inclusion relationships

Use Toroidal if:
- Data includes angular coordinates
- There are cyclic patterns in the data

Use Adaptive if:
- You don't know which geometry to use
- You have lots of data for geometry to emerge

Use Hierarchical if:
- Data has multi-scale structure
- You need to capture fine details

## Default Configuration

```python
geometry = ChristoffelLowRank(
    dim=512,
    rank=64,
    curvature_clamp=3.0,
    enable_trace_normalization=True
)
```

This configuration was chosen for a general balance between capacity and stability.

---

**Manifold Labs (Joaquín Stürtz)**
