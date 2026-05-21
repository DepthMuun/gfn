# Plugins - Fractal (Sub-Manifold Tunneling)

## What is Fractal Tunneling?

Fractal tunneling is a technique that refines the manifold evolution by adding a "micro-scale" dynamics layer. When the main manifold experiences high curvature, the fractal plugin creates a temporary sub-manifold to handle finer-scale physics.

Think of it as: "When things get complex, zoom in and think more carefully."

---

## The Problem

### High Curvature Regions

In regions where the velocity norm $\|v\|$ is large:
- The integrator may struggle with accuracy
- Large steps miss fine-scale structure
- Numerical errors accumulate

### Standard Solution
Reduce $dt$ globally - but this slows down the entire model.

### Fractal Solution
Detect high-curvature regions and add local refinement only where needed.

---

## How It Works

### Step 1: Curvature Detection

Estimate local curvature from velocity magnitude:

$$c = \|v\| = \sqrt{\sum_i v_i^2}$$

Where:
- $c$ = curvature estimate
- $v$ = velocity tensor
- Averaged across heads for stability

### Step 2: Tunnel Gate

Compute how much we should "tunnel" into the micro-manifold:

$$g = \sigma((c - \tau) \cdot s)$$

Where:
- $\sigma$ = sigmoid function
- $\tau$ = curvature threshold
- $s$ = slope (sharpness of transition)
- $g$ ∈ [0, 1] = tunnel gate value

**Behavior**:
- $c \ll \tau$: $g \approx 0$ (no tunneling)
- $c \approx \tau$: $g \approx 0.5$ (partial)
- $c \gg \tau$: $g \approx 1$ (full tunneling)

### Step 3: Micro-Manifold Evolution

If a micro-manifold exists:

$$(x_{micro}, v_{micro}) = \text{MicroManifold}(x, v)$$

This runs a smaller, faster integration on the sub-scale.

### Step 4: State Blending

Blend the original and micro-manifold states:

$$x_{out} = x + g \cdot \alpha \cdot (x_{micro} - x)$$
$$v_{out} = v + g \cdot \alpha \cdot (v_{micro} - v)$$

Where:
- $\alpha$ = blending strength (typically 0.1)
- $g$ = tunnel gate (0 to 1)

**Result**: Smooth transition from normal to refined evolution.

---

## Physical Interpretation

### Multi-Scale Physics

Think of the manifold as having structure at multiple scales:
- **Macro scale**: Main manifold evolution
- **Micro scale**: Fine details in high-curvature regions

The fractal plugin automatically switches between scales as needed.

### Energy Landscape Analogy

Imagine walking on a terrain:
- **Flat regions**: Walk normally (no tunneling)
- **Steep/cluttered regions**: Take smaller, careful steps (tunneling)

The tunnel gate $g$ determines how carefully to step.

---

## Mathematical Properties

### Curvature Estimation

$$c(x, v) = \frac{1}{H} \sum_{h=1}^H \|v_h\|$$

Average velocity norm across heads.

### Sigmoid Gate

$$g(c; \tau, s) = \frac{1}{1 + \exp(-s(c - \tau))}$$

Properties:
- Differentiable (smooth)
- Saturates at 0 and 1
- Threshold controlled by $\tau$

### Blending Formula

$$x_{new} = x + \alpha \cdot g(c) \cdot (x_{micro} - x)$$

This is a convex combination when $g \cdot \alpha \leq 1$.

---

## Parameters

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| Threshold | $\tau$ | 1.0 | When to start tunneling |
| Alpha | $\alpha$ | 0.1 | Blending strength |
| Slope | $s$ | 1.0 | Sharpness of transition |

### Tuning Guidelines

**Lower threshold** ($\tau$ = 0.5):
- More aggressive tunneling
- More compute overhead
- Better accuracy in moderate curvature

**Higher threshold** ($\tau$ = 2.0):
- Conservative tunneling
- Less overhead
- Only extreme curvature triggers it

**Higher alpha** ($\alpha$ = 0.2):
- Stronger micro-manifold influence
- More refinement
- Risk of instability

**Lower alpha** ($\alpha$ = 0.05):
- Weaker refinement
- More conservative
- Less benefit

---

## When to Use

**Use Fractal when:**
- Model shows instability in high-curvature regions
- You need higher accuracy in complex regions
- Computational budget allows extra overhead

**Don't use when:**
- Training is already stable
- Speed is critical (adds ~10-20% overhead)
- No micro-manifold is available

---

## Configuration

```python
physics = {
    'fractal': {
        'enabled': True,
        'threshold': 1.0,   # Curvature threshold
        'alpha': 0.1,       # Blending strength
        'slope': 1.0        # Gate sharpness
    }
}
```

---

*File: technical/0_architecture/math/plugins/fractal.md*
*Last Updated: 2026-04-02*
