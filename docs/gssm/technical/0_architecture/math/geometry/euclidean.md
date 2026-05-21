# Euclidean Geometry

## What is Euclidean Space?

Euclidean space is the familiar flat geometry of everyday experience: $\mathbb{R}^n$.

Unlike the torus, Euclidean space is:
- **Unbounded**: extends infinitely in all directions
- **Flat**: zero curvature everywhere
- **Linear**: straight lines are truly straight

Think of it as: "Regular flat space with no wrapping."

---

## Mathematical Definition

### Metric Tensor

The metric is simply the identity matrix:

$$g_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

In matrix form:
$$g = I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$$

### Distance

Standard Euclidean distance:
$$d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$$

---

## Christoffel Symbols

### Result

For Euclidean space, ALL Christoffel symbols are zero:

$$\Gamma^k_{ij} = 0 \quad \text{for all } i, j, k$$

### Physical Meaning

**No geometric force**:
$$a_{geo} = -\Gamma(x, v) = 0$$

The manifold exerts no "curvature force" on trajectories.

**Straight-line motion**:
Objects move in straight lines unless external forces act.

---

## Implications for GSSM

### Simpler Dynamics

Without geometric forces:
$$\frac{dv}{dt} = F_{ext} - \mu v$$

Just external force + friction.

### Unbounded State

Positions can grow indefinitely:
- No wrapping to $[-\pi, \pi]$
- State can explode if unchecked
- Requires stronger regularization

### When to Use

**Use Euclidean when**:
- Problem has natural linear structure
- Boundedness is not required
- Simplicity is preferred
- Working with continuous regression

**Don't use when**:
- Stability is critical
- Periodic patterns exist
- State explosion is a risk

---

## Comparison with Torus

| Property | Euclidean | Torus |
|----------|-----------|-------|
| Curvature | Flat (0) | Variable |
| Bounded | No | Yes |
| Geometric force | None | Present |
| Stability | Lower | Higher |
| Complexity | Simple | Rich |
| Wrapping | None | Periodic |

---

## Normalization Differences

### Position

**Euclidean**: Identity (no operation)
$$x_{norm} = x$$

**Torus**: Wrap to $[-\pi, \pi]$
$$x_{norm} = \arctan_2(\sin x, \cos x)$$

### Velocity

Same for both: clamp + RMS norm

---

## Why Torus is Default

Despite Euclidean simplicity, torus is the GSSM default because:

1. **Stability**: Bounded state prevents explosion
2. **Generalization**: Works for most problems
3. **Physics**: Curvature adds useful structure
4. **Safety**: Fail-safe behavior

---

*File: technical/0_architecture/math/geometry/euclidean.md*
*Last Updated: 2026-04-02*
