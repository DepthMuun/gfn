# Torus Geometry

## What is the Torus?

The torus is a doughnut-shaped manifold: $T^n = S^1 \times S^1 \times ... \times S^1$ (n circles).

In GSSM, we use a 2D torus model where position pairs $(\theta, \phi)$ represent angular coordinates:
- $\theta$ = angle around the tube (minor circle)
- $\phi$ = angle around the hole (major circle)

Think of it as: "A space where moving far enough brings you back to the start."

---

## Mathematical Definition

### Embedding in 3D

A torus can be embedded in 3D space:

$$x = (R + r\cos\theta)\cos\phi$$
$$y = (R + r\cos\theta)\sin\phi$$
$$z = r\sin\theta$$

Where:
- $R$ = major radius (distance from center to tube center)
- $r$ = minor radius (radius of the tube)

### Metric Tensor

The intrinsic metric (measuring distances ON the torus):

$$g = \begin{pmatrix} r^2 & 0 \\ 0 & (R + r\cos\theta)^2 \end{pmatrix}$$

**Properties**:
- Diagonal (no cross-terms)
- Position-dependent (second term varies with $\theta$)
- Always positive definite

---

## Christoffel Symbols

### Non-Zero Components

For the 2D torus:

$$\Gamma^\theta_{\phi\phi} = \frac{(R + r\cos\theta)\sin\theta}{r}$$

$$\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = -\frac{r\sin\theta}{R + r\cos\theta}$$

### Physical Meaning

These represent "fictitious forces" due to curvature:

**$\Gamma^\theta_{\phi\phi}$**: Centrifugal-like force
- When moving around $\phi$ (major circle)
- Pushes toward/away from inner tube

**$\Gamma^\phi_{\theta\phi}$**: Coriolis-like force
- Couples motion in $\theta$ and $\phi$
- Deflects trajectory

---

## Geodesic Force

### Computation

The geometric acceleration is:

$$a_{geo}^\theta = -\Gamma^\theta_{\phi\phi} \cdot v^\phi \cdot v^\phi$$
$$a_{geo}^\phi = -\Gamma^\phi_{\theta\phi} \cdot v^\theta \cdot v^\phi - \Gamma^\phi_{\phi\theta} \cdot v^\phi \cdot v^\theta$$

### Intuition

**Geodesics on torus**:
- "Straight lines" that wrap around
- Can be closed loops (periodic)
- Can be dense (never repeating)

**Types of geodesics**:
1. **Meridians**: Around tube ($\phi$ = constant)
2. **Parallels**: Around hole ($\theta$ = constant)
3. **General**: Combinations, winding patterns

---

## Learnable Radii

### Innovation

Unlike standard torus geometry, GSSM makes $R$ and $r$ **learnable parameters**.

**Why?**
- Geometry adapts to data
- Different heads can have different scales
- Emergent hierarchical structure

### Update Rule

$$R_{new} = R_{old} - \eta \frac{\partial L}{\partial R}$$
$$r_{new} = r_{old} - \eta \frac{\partial L}{\partial r}$$

After update, wrap to positive:
$$R = |R|, \quad r = |r|$$

---

## Position Wrapping

### The Problem

Angles are periodic: $\theta \equiv \theta + 2\pi$

During evolution, positions can drift outside $[-\pi, \pi]$.

### Solution

After each update:
$$\theta = \arctan_2(\sin\theta, \cos\theta)$$

**Properties**:
- Maps any angle to $[-\pi, \pi]$
- Differentiable
- Preserves periodicity

---

## Friction on Torus

### Adaptive Friction

Friction coefficient can vary with position:

$$\mu(\theta) = \mu_0 + \alpha \cdot \text{curvature}(\theta)$$

Higher friction in high-curvature regions (outer tube).

### Gate Mechanism

$$\mu_{eff} = \mu_{base} \cdot \sigma(W \cdot x + b)$$

State-dependent friction for stability.

---

## Why Use Torus?

### Advantages

| Property | Benefit |
|----------|---------|
| **Bounded** | No state explosion |
| **Periodic** | Natural for cyclic patterns |
| **Curved** | Rich geometric structure |
| **Compact** | Finite volume |

### Use Cases

- Language modeling (periodic patterns)
- Time series (seasonal/cyclic)
- Any bounded representation task

---

## Comparison with Euclidean

| Aspect | Torus | Euclidean |
|--------|-------|-----------|
| Bounded | ✓ Yes | ✗ No |
| Periodic | ✓ Yes | ✗ No |
| Curvature | ✓ Variable | ✗ Flat |
| Stability | ✓ Better | ✗ Can explode |
| Complexity | Higher | Lower |

---

*File: technical/0_architecture/math/geometry/torus.md*
*Last Updated: 2026-04-02*
