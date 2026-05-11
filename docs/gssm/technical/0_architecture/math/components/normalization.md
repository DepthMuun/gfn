# Manifold Normalization

## What is Manifold Normalization?

Manifold Normalization ensures that positions and velocities remain well-behaved during evolution. Unlike standard neural network normalization (like BatchNorm), manifold normalization is geometry-aware and respects the topology of the space.

Think of it as: "Keeping the manifold state within valid bounds while preserving geometric structure."

---

## Why Different Normalization for Position vs Velocity?

### Position Lives on Manifold
Position $x$ lives ON the manifold (the curved space itself).
- Torus: Periodic, bounded $[-\pi, \pi]$
- Euclidean: Unbounded

### Velocity Lives in Tangent Space
Velocity $v$ lives IN the tangent space (flat approximation at each point).
- Always Euclidean (even for curved manifolds)
- Can be normalized with standard techniques

---

## Position Normalization

### Torus Position Normalization

**Purpose**: Wrap position to valid torus coordinates $[-\pi, \pi]$.

**Formula**:
$$x_{norm} = \arctan_2(\sin(x), \cos(x))$$

**Why this works**:
- $\sin$ and $\cos$ are periodic with period $2\pi$
- $\arctan_2$ recovers the angle in $[-\pi, \pi]$
- Effectively "wraps" any value to the fundamental domain

**Example**:
- Input: $x = 3\pi$ (outside range)
- $\sin(3\pi) = 0$, $\cos(3\pi) = -1$
- $\arctan_2(0, -1) = \pi$
- Output: $\pi$ (equivalent position on torus)

### Euclidean Position Normalization

**Purpose**: Identity (no wrapping needed).

**Formula**:
$$x_{norm} = x$$

Euclidean space is unbounded, so no normalization is applied to position.

---

## Velocity Normalization

### Tangent Velocity Normalization

**Purpose**: Clamp and scale velocity in tangent space.

**Components**:

1. **Hard Clamping**: Prevent runaway velocity
   $$v_{clamped} = \text{clamp}(v, -v_{max}, +v_{max})$$

2. **RMS Normalization**: Scale to unit norm
   $$v_{norm} = \frac{v_{clamped}}{\sqrt{\frac{1}{d}\sum_i v_i^2 + \epsilon}}$$

**Combined**:
$$v_{out} = \text{RMSNorm}(\text{clamp}(v, \pm v_{max}))$$

### Metric-Aware Velocity Normalization

**Purpose**: Normalize using the Riemannian metric (geometry-aware).

**Riemannian Norm**:
The "true" velocity magnitude on a curved manifold is:
$$\|v\|_g^2 = v^T g(x) v$$

Where $g(x)$ is the metric tensor at position $x$.

**Normalization**:
1. Compute metric norm: $\|v\|_g = \sqrt{v^T g(x) v}$
2. Scale if exceeds limit: $v_{out} = v \cdot \min(1, \frac{v_{max}}{\|v\|_g})$

**Physical Meaning**:
- Standard norm $\|v\|$ = coordinate velocity
- Metric norm $\|v\|_g$ = physical velocity on manifold
- Ensures true physical speed limit

---

## The Normalization Registry

### Automatic Selection

The registry automatically selects appropriate normalization based on:
1. **Topology**: Torus vs Euclidean
2. **Variable type**: Position vs Velocity
3. **Geometry availability**: With or without metric tensor

### Selection Logic

```
IF is_velocity:
    IF geometry_available:
        RETURN MetricAwareVelocityNormalization
    ELSE:
        RETURN TangentVelocityNormalization
ELSE (is_position):
    IF topology == TORUS:
        RETURN TorusPositionNormalization
    ELSE:
        RETURN EuclideanPositionNormalization
```

---

## When Each Normalization Applies

| Variable | Topology | Normalization | Purpose |
|----------|----------|---------------|---------|
| Position | Torus | $\arctan_2(\sin, \cos)$ | Wrap to $[-\pi, \pi]$ |
| Position | Euclidean | Identity | No bounds |
| Velocity | Any | RMSNorm + Clamp | Prevent explosion |
| Velocity | With metric | Metric-aware | Physical speed limit |

---

## Physical Interpretation

### Position Wrapping (Torus)

Imagine walking on a circular track:
- After walking $2\pi$ radians, you're back at start
- $\arctan_2(\sin, \cos)$ computes your "true" position on the circle
- Handles multiple rotations correctly

### Velocity Clamping

Prevents "runaway" dynamics:
- Without limits: $v \to \infty$ (numerical explosion)
- With clamping: $|v| \leq v_{max}$ (stable)

### Metric-Aware Normalization

Like speed limits on curved roads:
- Flat highway: Speed limit is simple
- Mountain road: Speed limit varies by curvature
- Metric tensor $g(x)$ encodes local "road curvature"

---

## Mathematical Properties

### Torus Normalization is Idempotent

$$N(N(x)) = N(x)$$

Applying twice gives same result (already wrapped).

### Velocity Normalization is Bounded

$$\|v_{out}\| \leq v_{max}$$

Guaranteed by clamping.

### Metric Norm is Coordinate-Invariant

$$\|v\|_g^2 = v^T g v$$

Same physical velocity regardless of coordinate choice.

---

## Configuration

Normalization is typically enabled via stability config:

```python
physics = {
    'stability': {
        'enable_trace_normalization': True,  # Enable position norm
        'velocity_saturation': 10.0,          # Max velocity
    }
}
```

---

## Comparison with Standard Normalization

| Aspect | Standard (BatchNorm) | Manifold Normalization |
|--------|---------------------|------------------------|
| Operates on | Batch statistics | Individual samples |
| Learns | Scale/shift params | No params (or fixed) |
| Geometry aware | No | Yes |
| Purpose | Train stability | Physical validity |

---

*File: technical/0_architecture/math/components/normalization.md*
*Last Updated: 2026-04-02*
